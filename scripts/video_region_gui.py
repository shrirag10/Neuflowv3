"""Video region-flow tool.

Load a video, step through it frame by frame, drag a box, and compute optical
flow for that region only.

Two region modes, because they cost very different amounts and the difference
is the honest heart of this project:

  QUERY  -- run the full coarse pass on the whole frame (unavoidable: matching
            needs global context), then decode ONLY the pixels inside the box.
            Exact: identical to the full-frame result inside the box. Saves
            only decode time, which is the small part.

  CROP   -- feed only the box (plus a margin) through the entire pipeline.
            Cost scales with box area, so a quarter-size box is roughly 4x
            faster end to end. This is the mode that actually delivers "I only
            care about part of the image, so it should be much faster", but it
            is an approximation: the network loses the surrounding context that
            global matching uses, and motion that leaves the crop cannot be
            found. The margin slider trades speed for context.

Both timings are shown side by side so the trade-off is visible rather than
asserted.

Usage:
    python3 scripts/video_region_gui.py --video clip.mp4 \
        --checkpoint checkpoints/neuflowv3_mix/step_030000.pth
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QVBoxLayout, QSlider, QComboBox, QFileDialog,
                             QRubberBand, QStatusBar, QGroupBox, QGridLayout, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRect, QSize, QPoint

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils, flow_viz
from utils.load_model import my_load_weights, load_with_new_keys


# FlowEngine lives in flow_engine.py so the benchmarks can import it without Qt
from flow_engine import FlowEngine


# --------------------------------------------------------------------------- view
class FrameView(QLabel):
    """Displays a frame and lets the user drag a box on it."""

    def __init__(self, on_box):
        super().__init__()
        self.on_box = on_box
        self.setMouseTracking(True)
        self.rubber = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = None
        self.box = None            # (x, y, w, h) in image coords
        self.overlay = None        # RGB overlay to paint inside the box
        self.base = None
        self.scale = 1.0

    def set_frame(self, rgb):
        self.base = rgb
        h, w = rgb.shape[:2]
        qi = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qi))
        self.setFixedSize(w, h)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.box = None
            self.overlay = None
            self.rubber.hide()
            self.repaint_composite()
            return
        self.origin = ev.pos()
        self.rubber.setGeometry(QRect(self.origin, QSize()))
        self.rubber.show()

    def mouseMoveEvent(self, ev):
        if self.origin is not None:
            self.rubber.setGeometry(QRect(self.origin, ev.pos()).normalized())

    def mouseReleaseEvent(self, ev):
        if self.origin is None:
            return
        r = QRect(self.origin, ev.pos()).normalized()
        self.origin = None
        self.rubber.hide()
        if r.width() < 12 or r.height() < 12:
            return
        H, W = self.base.shape[:2]
        x0 = max(0, min(W - 1, r.x())); y0 = max(0, min(H - 1, r.y()))
        w = min(W - x0, r.width()); h = min(H - y0, r.height())
        self.box = (x0, y0, w, h)
        self.on_box(self.box)

    def repaint_composite(self):
        if self.base is None:
            return
        img = self.base.copy()
        if self.overlay is not None and self.box is not None:
            x0, y0, w, h = self.box
            img[y0:y0 + h, x0:x0 + w] = self.overlay
        H, W = img.shape[:2]
        qi = QImage(np.ascontiguousarray(img).data, W, H, 3 * W, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi)
        if self.box is not None:
            p = QPainter(pm)
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.drawRect(*self.box)
            p.end()
        self.setPixmap(pm)


# --------------------------------------------------------------------------- main
class RegionFlowGUI(QMainWindow):
    def __init__(self, engine, video=None):
        super().__init__()
        self.engine = engine
        self.cap = None
        self.frames = []
        self.idx = 0
        self.setWindowTitle('NeuFlow v3 -- region flow on video')

        self.view = FrameView(self.compute)
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # controls
        btn_open = QPushButton('Open video')
        btn_open.clicked.connect(self.open_video)
        self.btn_prev = QPushButton('< prev')
        self.btn_next = QPushButton('next >')
        self.btn_prev.clicked.connect(lambda: self.goto(self.idx - 1))
        self.btn_next.clicked.connect(lambda: self.goto(self.idx + 1))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.goto)

        self.mode = QComboBox()
        self.mode.addItems(['QUERY  (exact, decode box only)',
                            'CROP   (approx, whole pipeline on box)'])
        self.margin = QSlider(Qt.Horizontal)
        self.margin.setRange(0, 100); self.margin.setValue(25)
        self.margin.setFixedWidth(120)
        self.lbl_margin = QLabel('crop margin 25%')
        self.margin.valueChanged.connect(
            lambda v: self.lbl_margin.setText(f'crop margin {v}%'))
        self.chk_compare = QCheckBox('also time v2 full frame')
        self.chk_compare.setChecked(True)
        self.stride = QComboBox()
        self.stride.addItems(['stride 1 (every pixel)', 'stride 2 (4x cheaper)',
                              'stride 4 (16x cheaper)'])
        self.stride.setCurrentIndex(1)
        self.stride.currentIndexChanged.connect(
            lambda _: self.view.box and self.compute(self.view.box))

        top = QHBoxLayout()
        for wdg in (btn_open, self.btn_prev, self.btn_next):
            top.addWidget(wdg)
        top.addWidget(self.slider, 1)

        opts = QHBoxLayout()
        opts.addWidget(QLabel('mode:')); opts.addWidget(self.mode)
        opts.addWidget(self.lbl_margin); opts.addWidget(self.margin)
        opts.addWidget(QLabel('decode:')); opts.addWidget(self.stride)
        opts.addWidget(self.chk_compare); opts.addStretch(1)

        # measurement panel
        self.readout = QLabel('Drag a box on the frame to compute flow there.\n'
                              'Right-click clears.')
        self.readout.setFont(QFont('Monospace', 10))
        self.readout.setStyleSheet('background:#f4f4f4; padding:8px;')
        box = QGroupBox('measurements')
        bl = QVBoxLayout(); bl.addWidget(self.readout); box.setLayout(bl)

        lay = QVBoxLayout()
        lay.addLayout(top); lay.addLayout(opts)
        lay.addWidget(self.view, 1); lay.addWidget(box)
        c = QWidget(); c.setLayout(lay); self.setCentralWidget(c)

        if video:
            self.load_video(video)

    def stride_val(self):
        return [1, 2, 4][self.stride.currentIndex()]

    # ---- video ----
    def open_video(self):
        fn, _ = QFileDialog.getOpenFileName(self, 'Open video', '',
                                            'Video (*.mp4 *.avi *.mov *.mkv);;All (*)')
        if fn:
            self.load_video(fn)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok or len(frames) >= 400:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) < 2:
            self.status.showMessage(f'could not read 2+ frames from {path}')
            return
        self.frames = frames
        self.slider.setRange(0, len(frames) - 2)
        self.idx = 0
        h, w = frames[0].shape[:2]
        self.status.showMessage(f'{os.path.basename(path)}: {len(frames)} frames '
                                f'at {w}x{h} -- warming up GPU...')
        QApplication.processEvents()
        self.engine.warmup(h, w)
        self.status.showMessage(f'{os.path.basename(path)}: {len(frames)} frames '
                                f'at {w}x{h} -- ready (timings are steady-state)')
        self.show_frame()

    def goto(self, i):
        if not self.frames:
            return
        self.idx = max(0, min(len(self.frames) - 2, i))
        if self.slider.value() != self.idx:
            self.slider.blockSignals(True); self.slider.setValue(self.idx)
            self.slider.blockSignals(False)
        self.show_frame()

    def show_frame(self):
        self.view.overlay = None
        self.view.set_frame(self.frames[self.idx])
        self.view.repaint_composite()
        self.setWindowTitle(f'NeuFlow v3 -- region flow  |  frame {self.idx} -> {self.idx+1}')
        if self.view.box:
            self.compute(self.view.box)

    # ---- compute ----
    def compute(self, box):
        if not self.frames:
            return
        x0, y0, w, h = box
        self.view.box = box          # so overlay + outline render for
                                     # programmatic calls too, not just mouse drags
        i0, i1 = self.frames[self.idx], self.frames[self.idx + 1]
        H, W = i0.shape[:2]
        frac = (w * h) / float(H * W)
        lines = [f'box  {w}x{h} at ({x0},{y0})   =  {frac*100:5.1f}% of the frame']

        if self.mode.currentIndex() == 0:
            st = self.engine.coarse(i0, i1, key=(id(self.frames), self.idx))
            f, b, ms = self.engine.query_region(st, x0, y0, w, h, stride=self.stride_val())
            cached = (self.engine.state_key == (id(self.frames), self.idx))
            lines += [
                f'QUERY mode (exact)',
                f'  coarse pass, whole frame : {self.engine.coarse_ms:7.1f} ms'
                f'{"  (cached, not re-run)" if cached else ""}',
                f'  decode {(w//self.stride_val())*(h//self.stride_val()):>7,} pts'
                f' (stride {self.stride_val()}): {ms:7.1f} ms',
                f'  total this frame         : {self.engine.coarse_ms + ms:7.1f} ms',
                f'  another box, same frame  : {ms:7.1f} ms  (coarse is reused)',
            ]
            if b is not None:
                lines.append(f'  predicted error b        : median {np.median(b):.2f} px')
        else:
            m = self.margin.value() / 100.0
            f, c_ms, d_ms, cropped_px = self.engine.crop_region(
                i0, i1, x0, y0, w, h, m, stride=self.stride_val())
            self.engine.coarse(i0, i1, key=(id(self.frames), self.idx))
            full_c = self.engine.coarse_ms
            lines += [
                f'CROP mode (approximate: no context outside the crop)',
                f'  coarse on crop+{int(m*100):>3}%      : {c_ms:7.1f} ms'
                f'   ({cropped_px/float(H*W)*100:.0f}% of frame area)',
                f'  decode box (stride {self.stride_val()})     : {d_ms:7.1f} ms',
                f'  total this frame         : {c_ms + d_ms:7.1f} ms',
                f'  --- like-for-like: coarse pass only ---',
                f'  coarse, full frame       : {full_c:7.1f} ms',
                f'  coarse, cropped          : {c_ms:7.1f} ms'
                f'   -> {full_c/max(c_ms,1e-6):.2f}x',
                f'  (decode cost is the same either way; only the coarse pass shrinks)',
            ]

        if self.chk_compare.isChecked() and self.engine.v2 is not None:
            _, v2ms = self.engine.full_frame_v2(i0, i1)
            lines.append(f'v2 dense, whole frame      : {v2ms:7.1f} ms  (its only mode)')

        mag = np.linalg.norm(f, axis=-1)
        lines.append(f'flow in box: mean {mag.mean():.2f} px, max {mag.max():.2f} px')

        self.view.overlay = flow_viz.flow_to_image(f)
        self.view.repaint_composite()
        self.readout.setText('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=None)
    ap.add_argument('--checkpoint', default='checkpoints/neuflowv3_mix/step_030000.pth')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--head', default='convex', choices=['regress', 'convex'])
    ap.add_argument('--uncertainty', action='store_true')
    ap.add_argument('--selftest', action='store_true',
                    help='offscreen: synthesize a clip, run both modes, save a PNG')
    args = ap.parse_args()

    if args.selftest:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    app = QApplication(sys.argv)
    eng = FlowEngine(args.checkpoint, args.v2_checkpoint, args.head, args.uncertainty)
    gui = RegionFlowGUI(eng, args.video)

    if args.selftest:
        if not gui.frames:
            # synthesize a moving-texture clip so the test needs no video file
            rng = np.random.RandomState(0)
            tex = (rng.rand(300, 700, 3) * 90 + 60).astype(np.uint8)
            tex = cv2.GaussianBlur(tex, (0, 0), 4)
            for _ in range(14):      # shapes give the eye something to track
                c = tuple(int(v) for v in rng.randint(40, 255, 3))
                x, y = rng.randint(0, 690), rng.randint(0, 290)
                if rng.rand() > .5:
                    cv2.circle(tex, (x, y), rng.randint(12, 34), c, -1)
                else:
                    cv2.rectangle(tex, (x, y), (x + rng.randint(20, 60),
                                                y + rng.randint(20, 60)), c, -1)
            gui.frames = [np.roll(tex, i * 3, axis=1)[:, :640].copy() for i in range(4)]
            gui.slider.setRange(0, 2)
            eng.warmup(*gui.frames[0].shape[:2])
            gui.show_frame()
        for mode in (0, 1):
            gui.mode.setCurrentIndex(mode)
            # first call pays one-time CUDA autotuning for any new tensor shape
            # (measured at 50-90 ms for a fresh crop size); report steady state
            gui.compute((160, 60, 260, 150))
            gui.compute((160, 60, 260, 150))
            print(f'--- mode {mode} (steady state) ---')
            print(gui.readout.text())
        os.makedirs('results/visuals', exist_ok=True)
        gui.view.pixmap().save('results/visuals/region_gui_frame.png')
        gui.resize(980, 620)
        QApplication.processEvents()
        gui.grab().save('results/visuals/region_gui_window.png')
        print('saved results/visuals/region_gui_frame.png and region_gui_window.png')
        return

    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
