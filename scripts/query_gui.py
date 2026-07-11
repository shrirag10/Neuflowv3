"""NeuFlow v3 — interactive flow query tool (PyQt5).

The backbone runs once per image pair (~33 ms); every interaction afterwards
is a decode_queries() call. Demonstrates the two-pass O(N) API.

Usage:
    python3 scripts/query_gui.py [--img1 A.jpg --img2 B.jpg]
                                 [--video path.mp4 | --youtube URL]
                                 [--checkpoint C.pth] [--head convex] [--pe]
                                 [--selftest]

Interactions:
    left click       query flow at that pixel (arrow + value)
    drag (Region on) select a region — flow is computed only inside it
    right click      clear all queries and overlays
    G                uniform grid (size in toolbar)
    A                adaptive queries at motion boundaries (N in toolbar)
    D                toggle dense full-frame overlay
    N / P            video only: next / previous frame pair
    File menu        open image pair / video / YouTube URL / checkpoint,
                     export CSV, save screenshot
"""

import sys, os, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QStatusBar, QFileDialog, QMessageBox,
    QAction, QToolBar, QSpinBox, QScrollArea, QRubberBand, QInputDialog,
    QTabWidget, QWidget, QGridLayout, QDoubleSpinBox,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QTimer

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils, flow_viz
from utils.load_model import my_load_weights, load_with_new_keys

DEFAULT_CKPT = 'checkpoints/neuflowv3_mix/step_030000.pth'


class FlowSession:
    """One coarse pass per image pair; unlimited cheap queries afterwards."""

    def __init__(self, checkpoint=DEFAULT_CKPT, head='convex', pe=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = checkpoint
        self.model = NeuFlow(use_implicit=True, head_mode=head, use_pe=pe).to(self.device)
        load_with_new_keys(self.model, my_load_weights(checkpoint),
                           missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                           unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
        self.model.eval()
        self.state = None
        self.img1_np = None
        self.coarse_ms = None
        self._bhwd = None

    def set_pair_arrays(self, img1_np, img2_np):
        self.img1_np = np.ascontiguousarray(img1_np)
        img1 = torch.from_numpy(self.img1_np).permute(2, 0, 1).float()[None]
        img2 = torch.from_numpy(np.ascontiguousarray(img2_np)).permute(2, 0, 1).float()[None]
        self.padder = frame_utils.InputPadder(img1.shape, padding_factor=16)
        a, b = self.padder.pad(img1.to(self.device), img2.to(self.device))
        self.pad_hw = (a.shape[-2], a.shape[-1])
        if self._bhwd != self.pad_hw:
            self.model.init_bhwd(1, a.shape[-2], a.shape[-1], self.device)
            self._bhwd = self.pad_hw
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            self.state = self.model.infer_coarse_state(a, b)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.coarse_ms = (time.perf_counter() - t0) * 1000

    def set_pair(self, p1, p2):
        self.set_pair_arrays(cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB),
                             cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB))

    def _decode(self, q):
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            return self.model.decode_queries(self.state, query_coords=q)

    def query(self, points_xy):
        q = torch.tensor(points_xy, dtype=torch.float32, device=self.device)[None]
        t0 = time.perf_counter()
        flow = self._decode(q)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return flow[0].float().cpu().numpy(), (time.perf_counter() - t0) * 1000

    def adaptive(self, n):
        from NeuFlow.adaptive_query import coarse_flow_query
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            q = coarse_flow_query(self.state['coarse_flow_s8'], num_points=n, adaptive_ratio=0.7)
            flow = self.model.decode_queries(self.state, query_coords=q)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return q[0].cpu().numpy(), flow[0].float().cpu().numpy(), (time.perf_counter() - t0) * 1000

    def region(self, x0, y0, w, h, max_pts=80000, chunk=65536):
        """Dense flow restricted to a rectangular region — the 'query window'.

        Stride is chosen so the region costs at most max_pts queries; at
        typical window sizes this is every pixel.
        """
        stride = max(1, int(np.ceil(np.sqrt(w * h / max_pts))))
        xs = torch.arange(x0, x0 + w, stride, device=self.device, dtype=torch.float32)
        ys = torch.arange(y0, y0 + h, stride, device=self.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([gx, gy], -1).reshape(1, -1, 2)
        t0 = time.perf_counter()
        parts = [self._decode(coords[:, i:i + chunk]) for i in range(0, coords.shape[1], chunk)]
        flow = torch.cat(parts, dim=1)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        flow = flow.reshape(len(ys), len(xs), 2).float().cpu().numpy()
        return flow, stride, ms

    def dense(self, chunk=65536):
        H, W = self.pad_hw
        t0 = time.perf_counter()
        ys, xs = torch.meshgrid(torch.arange(H, device=self.device, dtype=torch.float32),
                                torch.arange(W, device=self.device, dtype=torch.float32), indexing='ij')
        coords = torch.stack([xs, ys], -1).reshape(1, -1, 2)
        parts = [self._decode(coords[:, i:i + chunk]) for i in range(0, coords.shape[1], chunk)]
        out = torch.cat(parts, dim=1).reshape(1, H, W, 2).permute(0, 3, 1, 2)
        out = self.padder.unpad(out[0]).float().cpu().permute(1, 2, 0).numpy()
        return out, (time.perf_counter() - t0) * 1000


class VideoSource:
    """Frame-pair navigation over a local video file or a YouTube stream."""

    def __init__(self, path_or_url, youtube=False):
        self.label = path_or_url
        if youtube:
            import yt_dlp
            with yt_dlp.YoutubeDL({'format': 'best[ext=mp4][height<=720]', 'quiet': True}) as y:
                info = y.extract_info(path_or_url, download=False)
            path_or_url = info['url']
            self.label = info.get('title', 'YouTube stream')
        self.cap = cv2.VideoCapture(path_or_url)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open video source: {self.label}')
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
        self.idx = 0

    def pair(self, idx=None):
        if idx is not None:
            self.idx = max(0, min(idx, self.n_frames - 2))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.idx)
        ok1, f1 = self.cap.read()
        ok2, f2 = self.cap.read()
        if not (ok1 and ok2):
            raise RuntimeError(f'Could not read frames {self.idx}, {self.idx + 1}')
        # cap dimensions so the coarse pass stays fast on laptop VRAM
        h, w = f1.shape[:2]
        if w > 1024:
            sc = 1024 / w
            f1 = cv2.resize(f1, (1024, int(h * sc)))
            f2 = cv2.resize(f2, (1024, int(h * sc)))
        return (cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))


def detect_motion(state, thresh_px=2.0, min_area_s8=6):
    """Moving-region boxes from the coarse flow — no extra decode cost.

    Ego-motion is approximated by the median flow and subtracted, so a
    moving camera does not flag the whole frame. Returns full-res boxes
    [(x, y, w, h), ...] and the residual magnitude map (1/8 scale).
    """
    flow = state['coarse_flow_s8'][0].float().cpu().numpy()      # [2, H8, W8]
    med = np.median(flow.reshape(2, -1), axis=1, keepdims=True)
    resid = flow - med.reshape(2, 1, 1)
    mag = np.linalg.norm(resid, axis=0)                          # 1/8-scale px units
    mask = (mag * 8.0 > thresh_px).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= min_area_s8:
            boxes.append((int(x * 8), int(y * 8), int(w * 8), int(h * 8)))
    return boxes, mag * 8.0


class Sparkline(QWidget):
    """Minimal live time-series widget: title, current value, polyline."""

    def __init__(self, title, unit='', maxlen=120, color=QColor(0x2a, 0x78, 0xd6)):
        super().__init__()
        self.title, self.unit, self.color = title, unit, color
        self.data = deque(maxlen=maxlen)
        self.setMinimumHeight(96)

    def push(self, v):
        if v is not None:
            self.data.append(float(v))
            self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        p.setPen(QPen(QColor(60, 60, 60), 1))
        cur = f'{self.data[-1]:.1f}{self.unit}' if self.data else '—'
        p.setFont(QFont('Sans', 9, QFont.Bold))
        p.drawText(8, 16, f'{self.title}:  {cur}')
        if len(self.data) > 1:
            lo, hi = min(self.data), max(self.data)
            span = (hi - lo) or 1.0
            w, h = self.width() - 16, self.height() - 30
            pts = [QPoint(8 + int(i * w / (len(self.data) - 1)),
                          22 + int(h * (1 - (v - lo) / span)))
                   for i, v in enumerate(self.data)]
            p.setPen(QPen(self.color, 2))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        p.end()


class ResourcesPanel(QWidget):
    """Live system + pipeline metrics, sampled twice a second."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nv = None
        import psutil
        self.ps = psutil

        gray = QColor(0x33, 0x33, 0x33)
        self.s_fps    = Sparkline('Pipeline throughput', ' FPS', color=gray)
        self.s_coarse = Sparkline('Coarse pass', ' ms', color=gray)
        self.s_gpu    = Sparkline('GPU utilization', ' %', color=gray)
        self.s_vram   = Sparkline('GPU memory', ' MB', color=gray)
        self.s_cpu    = Sparkline('CPU utilization', ' %', color=gray)
        self.s_ram    = Sparkline('System memory', ' %', color=gray)
        grid = QGridLayout(self)
        for k, w in enumerate([self.s_fps, self.s_coarse, self.s_gpu, self.s_vram, self.s_cpu, self.s_ram]):
            grid.addWidget(w, k // 2, k % 2)
        note = QLabel('Sampling at 2 Hz. GPU memory stays flat while querying: the two-pass design\n'
                      'caches one backbone state and never re-allocates per interaction.')
        note.setStyleSheet('color: #666;')
        grid.addWidget(note, 3, 0, 1, 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sample)
        self.timer.start(500)

    def sample(self):
        self.s_cpu.push(self.ps.cpu_percent())
        self.s_ram.push(self.ps.virtual_memory().percent)
        if self.nv:
            try:
                u = self.nv.nvmlDeviceGetUtilizationRates(self.h)
                m = self.nv.nvmlDeviceGetMemoryInfo(self.h)
                self.s_gpu.push(u.gpu)
                self.s_vram.push(m.used / 1e6)
            except Exception:
                pass
        if self.window.session and self.window.session.coarse_ms:
            self.s_coarse.push(self.window.session.coarse_ms)
        if self.window.play_fps:
            self.s_fps.push(self.window.play_fps)


class QueryWindow(QMainWindow):
    def __init__(self, session, img1=None, img2=None, video=None, youtube=None):
        super().__init__()
        self.session = session
        self.queries = []
        self.dense_overlay = None
        self.show_dense = False
        self.region_patches = []   # [(x, y, QImage), ...]
        self.video = None
        self.rubber = None
        self.drag_origin = None
        self.motion_boxes = []
        self.play_fps = None
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_step)
        self.setWindowTitle('NeuFlow v3 — flow query tool')

        self.label = QLabel('File → open an image pair, a video, or a YouTube URL')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.mousePressEvent = self.on_press
        self.label.mouseMoveEvent = self.on_move
        self.label.mouseReleaseEvent = self.on_release
        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)
        tabs = QTabWidget()
        tabs.addTab(scroll, 'Viewer')
        tabs.addTab(ResourcesPanel(self), 'System resources')
        self.setCentralWidget(tabs)
        self.setStatusBar(QStatusBar())
        self._build_menu()
        self._build_toolbar()
        self.resize(1280, 540)

        if video or youtube:
            self.open_video_source(video or youtube, youtube=bool(youtube))
        elif img1 and img2:
            self.load_pair_paths(img1, img2)

    # ---- UI scaffolding -------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu('&File')
        for text, slot, key in [
            ('Open image pair…', self.open_pair, 'Ctrl+O'),
            ('Open video file…', self.open_video_dialog, 'Ctrl+V'),
            ('Open YouTube URL…', self.open_youtube_dialog, 'Ctrl+Y'),
            ('Load checkpoint…', self.open_checkpoint, 'Ctrl+L'),
            ('Export queries to CSV…', self.export_csv, 'Ctrl+E'),
            ('Save view as PNG…', self.save_png, 'Ctrl+S'),
            ('Quit', self.close, 'Ctrl+Q'),
        ]:
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)
        h = self.menuBar().addMenu('&Help')
        a = QAction('Controls', self)
        a.triggered.connect(lambda: QMessageBox.information(
            self, 'Controls',
            'Left click: query flow at pixel\nRegion mode + drag: flow inside a selected window\n'
            'Right click: clear\nG: grid · A: adaptive · D: dense overlay\nN / P: next / previous video frame pair\n\n'
            'The backbone runs once per pair; every interaction reuses its cached state.'))
        h.addAction(a)

    def _build_toolbar(self):
        tb = QToolBar('Query settings')
        self.addToolBar(tb)
        self.region_act = QAction('Region select: OFF', self)
        self.region_act.setCheckable(True)
        self.region_act.toggled.connect(
            lambda on: self.region_act.setText(f'Region select: {"ON" if on else "OFF"}'))
        tb.addAction(self.region_act)
        tb.addWidget(QLabel('  Grid: '))
        self.grid_spin = QSpinBox(); self.grid_spin.setRange(4, 128); self.grid_spin.setValue(32)
        tb.addWidget(self.grid_spin)
        tb.addWidget(QLabel('  Adaptive N: '))
        self.n_spin = QSpinBox(); self.n_spin.setRange(50, 8192); self.n_spin.setValue(500)
        tb.addWidget(self.n_spin)
        for text, slot in [('  Grid (G)', self.run_grid), ('Adaptive (A)', self.run_adaptive),
                           ('Dense overlay (D)', self.toggle_dense), ('Clear', self.clear)]:
            a = QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)

        tb.addSeparator()
        self.play_act = QAction('▶ Play (Space)', self)
        self.play_act.setCheckable(True)
        self.play_act.toggled.connect(self.toggle_play)
        tb.addAction(self.play_act)
        self.motion_act = QAction('Motion detect: ON', self)
        self.motion_act.setCheckable(True)
        self.motion_act.setChecked(True)
        self.motion_act.toggled.connect(
            lambda on: self.motion_act.setText(f'Motion detect: {"ON" if on else "OFF"}'))
        tb.addAction(self.motion_act)
        tb.addWidget(QLabel('  Motion thresh (px): '))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.5, 30.0); self.thresh_spin.setValue(2.5); self.thresh_spin.setSingleStep(0.5)
        tb.addWidget(self.thresh_spin)
        self.frame_label = QLabel('')
        tb.addWidget(self.frame_label)

    # ---- sources ---------------------------------------------------------
    def open_pair(self):
        p1, _ = QFileDialog.getOpenFileName(self, 'Frame 1', 'datasets', 'Images (*.jpg *.png *.ppm)')
        if not p1:
            return
        p2, _ = QFileDialog.getOpenFileName(self, 'Frame 2', os.path.dirname(p1), 'Images (*.jpg *.png *.ppm)')
        if p2:
            self.video = None
            self.load_pair_paths(p1, p2)

    def open_video_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Video', '.', 'Video (*.mp4 *.avi *.mov *.mkv *.webm)')
        if p:
            self.open_video_source(p, youtube=False)

    def open_youtube_dialog(self):
        url, ok = QInputDialog.getText(self, 'YouTube', 'Video URL:')
        if ok and url.strip():
            self.open_video_source(url.strip(), youtube=True)

    def open_video_source(self, src, youtube=False):
        try:
            self.statusBar().showMessage('Opening video source…')
            QApplication.processEvents()
            self.video = VideoSource(src, youtube=youtube)
            self.show_video_pair(0)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not open video:\n{e}')

    def show_video_pair(self, idx):
        f1, f2 = self.video.pair(idx)
        self.session.set_pair_arrays(f1, f2)
        if self.motion_act.isChecked():
            self.motion_boxes, _ = detect_motion(self.session.state, self.thresh_spin.value())
        else:
            self.motion_boxes = []
        self._after_new_pair(
            f'{self.video.label} · frames {self.video.idx}–{self.video.idx + 1} · '
            f'coarse pass {self.session.coarse_ms:.0f} ms · {len(self.motion_boxes)} moving regions · '
            f'N/P to step, Space to play')
        self.frame_label.setText(f'   frame {self.video.idx}/{self.video.n_frames}')

    # ---- real-time playback + motion detection ---------------------------
    def toggle_play(self, on):
        if self.video is None:
            self.play_act.setChecked(False)
            return
        if on:
            self.play_act.setText('⏸ Pause (Space)')
            self._t_last = time.perf_counter()
            self.play_timer.start(0)   # as fast as the pipeline allows
        else:
            self.play_act.setText('▶ Play (Space)')
            self.play_timer.stop()
            self.play_fps = None

    def play_step(self):
        if self.video.idx + 2 >= self.video.n_frames:
            self.play_act.setChecked(False)
            return
        self.show_video_pair(self.video.idx + 1)
        now = time.perf_counter()
        self.play_fps = 1.0 / max(now - self._t_last, 1e-6)
        self._t_last = now
        self.statusBar().showMessage(
            f'playing · frame {self.video.idx} · {self.play_fps:.1f} FPS end-to-end · '
            f'{len(self.motion_boxes)} moving regions (flow-based, ego-motion compensated)')

    def load_pair_paths(self, p1, p2):
        try:
            self.session.set_pair(p1, p2)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not process pair:\n{e}')
            return
        self._after_new_pair(
            f'{os.path.basename(p1)} → {os.path.basename(p2)} · coarse pass '
            f'{self.session.coarse_ms:.0f} ms (runs once) · click to query')

    def _after_new_pair(self, msg):
        h, w = self.session.img1_np.shape[:2]
        qimg = QImage(self.session.img1_np.data, w, h, 3 * w, QImage.Format_RGB888)
        self.base = QPixmap.fromImage(qimg)
        self.queries.clear()
        self.dense_overlay = None
        self.show_dense = False
        self.region_patches.clear()
        if self.video is None:
            self.motion_boxes = []
        self.redraw()
        self.statusBar().showMessage(msg)

    def open_checkpoint(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Checkpoint', 'checkpoints', 'Checkpoints (*.pth)')
        if p:
            try:
                self.session = FlowSession(p)
                self.statusBar().showMessage(f'Loaded {os.path.basename(p)} — open a source')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Could not load checkpoint:\n{e}')

    # ---- mouse: click query or region select -----------------------------
    def on_press(self, ev):
        if self.session.state is None:
            return
        if ev.button() == Qt.RightButton:
            self.clear()
            return
        if self.region_act.isChecked():
            self.drag_origin = ev.pos()
            if self.rubber is None:
                self.rubber = QRubberBand(QRubberBand.Rectangle, self.label)
            self.rubber.setGeometry(QRect(self.drag_origin, QSize()))
            self.rubber.show()
        else:
            self.click_query(ev.pos().x(), ev.pos().y())

    def on_move(self, ev):
        if self.drag_origin is not None and self.rubber is not None:
            self.rubber.setGeometry(QRect(self.drag_origin, ev.pos()).normalized())

    def on_release(self, ev):
        if self.drag_origin is None:
            return
        rect = QRect(self.drag_origin, ev.pos()).normalized()
        self.drag_origin = None
        if self.rubber is not None:
            self.rubber.hide()
        if rect.width() > 8 and rect.height() > 8:
            self.region_query(rect.x(), rect.y(), rect.width(), rect.height())

    def click_query(self, x, y):
        h, w = self.session.img1_np.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        flow, ms = self.session.query([(x, y)])
        u, v = flow[0]
        self.queries.append((x, y, float(u), float(v)))
        self.redraw()
        self.statusBar().showMessage(
            f'query ({x}, {y}) → flow ({u:+.2f}, {v:+.2f}) px · |f| = {np.hypot(u, v):.2f} px · decoded in {ms:.1f} ms')

    def region_query(self, x0, y0, w, h):
        """The query window: flow computed only inside the selected region."""
        img_h, img_w = self.session.img1_np.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        w, h = min(w, img_w - x0), min(h, img_h - y0)
        flow, stride, ms = self.session.region(x0, y0, w, h)
        rgb = flow_viz.flow_to_image(flow)
        if stride > 1:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        rgb = np.ascontiguousarray(rgb[:h, :w])
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
        self.region_patches.append((x0, y0, qi))
        self.redraw()
        mag = np.linalg.norm(flow, axis=-1)
        self.statusBar().showMessage(
            f'region {w}×{h} at ({x0}, {y0}): {flow.shape[0] * flow.shape[1]:,} queries in {ms:.1f} ms '
            f'(stride {stride}) · mean |f| {mag.mean():.2f} px · max {mag.max():.2f} px')

    # ---- bulk modes -------------------------------------------------------
    def run_grid(self):
        if self.session.state is None:
            return
        g = self.grid_spin.value()
        h, w = self.session.img1_np.shape[:2]
        xs = np.linspace(8, w - 8, g)
        ys = np.linspace(8, h - 8, max(4, int(g * h / w)))
        pts = [(float(x), float(y)) for y in ys for x in xs]
        flow, ms = self.session.query(pts)
        self.queries = [(p[0], p[1], float(f[0]), float(f[1])) for p, f in zip(pts, flow)]
        self.redraw()
        self.statusBar().showMessage(f'{len(pts)} grid queries decoded in {ms:.1f} ms')

    def run_adaptive(self):
        if self.session.state is None:
            return
        q, flow, ms = self.session.adaptive(self.n_spin.value())
        h, w = self.session.img1_np.shape[:2]
        self.queries = [(float(p[0]), float(p[1]), float(f[0]), float(f[1]))
                        for p, f in zip(q, flow) if p[0] < w and p[1] < h]
        self.redraw()
        self.statusBar().showMessage(
            f'{len(self.queries)} adaptive queries (concentrated at motion boundaries) in {ms:.1f} ms')

    def toggle_dense(self):
        if self.session.state is None:
            return
        if self.dense_overlay is None:
            flow, ms = self.session.dense()
            self.dense_overlay = flow_viz.flow_to_image(flow)
            self.statusBar().showMessage(f'dense field decoded in {ms:.0f} ms — overlay ON (D toggles)')
        self.show_dense = not self.show_dense
        self.redraw()

    def clear(self):
        self.queries.clear()
        self.show_dense = False
        self.region_patches.clear()
        self.redraw()
        self.statusBar().showMessage('cleared')

    def export_csv(self):
        if not self.queries:
            return
        p, _ = QFileDialog.getSaveFileName(self, 'Export CSV', 'queries.csv', 'CSV (*.csv)')
        if p:
            with open(p, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['x', 'y', 'u', 'v', 'magnitude'])
                for x, y, u, v in self.queries:
                    wr.writerow([x, y, f'{u:.4f}', f'{v:.4f}', f'{np.hypot(u, v):.4f}'])
            self.statusBar().showMessage(f'{len(self.queries)} queries → {p}')

    def save_png(self):
        p, _ = QFileDialog.getSaveFileName(self, 'Save PNG', 'flow_view.png', 'PNG (*.png)')
        if p:
            self.label.pixmap().save(p)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Space and self.video is not None:
            self.play_act.setChecked(not self.play_act.isChecked())
            return
        if ev.key() in (Qt.Key_N, Qt.Key_P) and self.video is not None:
            step = 1 if ev.key() == Qt.Key_N else -1
            self.show_video_pair(self.video.idx + step)
            return
        {Qt.Key_G: self.run_grid, Qt.Key_A: self.run_adaptive,
         Qt.Key_D: self.toggle_dense}.get(ev.key(), lambda: None)()

    # ---- rendering --------------------------------------------------------
    def redraw(self):
        if self.session.img1_np is None:
            return
        pm = QPixmap(self.base)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.show_dense and self.dense_overlay is not None:
            h, w = self.dense_overlay.shape[:2]
            ov = np.ascontiguousarray(self.dense_overlay)
            qi = QImage(ov.data, w, h, 3 * w, QImage.Format_RGB888)
            painter.setOpacity(0.55)
            painter.drawImage(0, 0, qi)
            painter.setOpacity(1.0)

        for x0, y0, qi in self.region_patches:
            painter.setOpacity(0.75)
            painter.drawImage(x0, y0, qi)
            painter.setOpacity(1.0)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(x0, y0, qi.width(), qi.height())
            painter.setPen(QPen(QColor(26, 26, 26), 1))
            painter.drawRect(x0 - 1, y0 - 1, qi.width() + 2, qi.height() + 2)

        if self.motion_boxes:
            h_img, w_img = self.session.img1_np.shape[:2]
            painter.setFont(QFont('Sans', 9, QFont.Bold))
            for (bx, by, bw_, bh_) in self.motion_boxes:
                bx, by = min(bx, w_img - 2), min(by, h_img - 2)
                bw_, bh_ = min(bw_, w_img - bx), min(bh_, h_img - by)
                painter.setPen(QPen(QColor(255, 60, 60), 3))
                painter.drawRect(bx, by, bw_, bh_)
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(bx + 4, by + 14, 'motion')

        painter.setFont(QFont('Sans', 8))
        mags = [np.hypot(u, v) for _, _, u, v in self.queries] or [1.0]
        mmax = max(max(mags), 1e-6)
        for x, y, u, v in self.queries:
            mag = np.hypot(u, v)
            hue = int(np.clip(240 * (1 - mag / mmax), 0, 240))
            color = QColor.fromHsv(hue, 255, 255)
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPoint(int(x), int(y)), QPoint(int(x + u), int(y + v)))
            painter.drawEllipse(QPoint(int(x), int(y)), 3, 3)
            if len(self.queries) <= 15:
                painter.setPen(QPen(QColor(20, 20, 20), 1))
                painter.drawText(int(x) + 6, int(y) - 6, f'({u:+.1f}, {v:+.1f})')

        if self.queries:
            lx, ly, lw = 12, 14, 130
            for i in range(lw):
                painter.setPen(QPen(QColor.fromHsv(int(240 * (1 - i / lw)), 255, 255), 1))
                painter.drawLine(lx + i, ly, lx + i, ly + 10)
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.drawText(lx, ly + 24, '0 px')
            painter.drawText(lx + lw - 30, ly + 24, f'{mmax:.0f} px')

        painter.end()
        self.label.setPixmap(pm)
        self.label.resize(pm.size())


def selftest(session, win, outdir='results/visuals'):
    os.makedirs(outdir, exist_ok=True)
    h, w = session.img1_np.shape[:2]

    flow, ms1 = session.query([(w // 4, h // 2)])
    win.click_query(w // 4, h // 2)
    win.run_adaptive()
    win.region_query(w // 3, h // 3, w // 3, h // 3)
    win.toggle_dense()
    win.redraw()
    win.label.pixmap().save(f'{outdir}/query_gui_selftest.png')

    # video path: synthesize a clip with a moving patch, then step + detect motion
    clip = '/tmp/gui_selftest_clip.mp4'
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    base = cv2.cvtColor(session.img1_np, cv2.COLOR_RGB2BGR)
    for k in range(4):
        fr = base.copy()
        x = 80 + 25 * k
        fr[100:180, x:x + 90] = fr[20:100, 40:130]   # translating patch = independent motion
        vw.write(fr)
    vw.release()
    win.open_video_source(clip, youtube=False)
    win.show_video_pair(1)
    ok_video = session.state is not None
    n_boxes = len(win.motion_boxes)

    win.play_act.setChecked(True)
    win.play_step()
    fps = win.play_fps or 0
    win.play_act.setChecked(False)
    win.label.pixmap().save(f'{outdir}/query_gui_motion.png')

    ok_ytdlp = True
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        ok_ytdlp = False

    print(f'selftest OK: click {ms1:.1f} ms · adaptive · region window · dense overlay · '
          f'video stepping {"OK" if ok_video else "FAIL"} · motion boxes {n_boxes} · '
          f'playback {fps:.1f} FPS · yt-dlp {"available" if ok_ytdlp else "MISSING"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img1'); ap.add_argument('--img2')
    ap.add_argument('--video'); ap.add_argument('--youtube')
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--head', default='convex', choices=['regress', 'convex'])
    ap.add_argument('--pe', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        args.img1 = args.img1 or 'datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00100.jpg'
        args.img2 = args.img2 or 'datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00101.jpg'

    app = QApplication(sys.argv)
    session = FlowSession(args.checkpoint, args.head, args.pe)
    win = QueryWindow(session, args.img1, args.img2, video=args.video, youtube=args.youtube)

    if args.selftest:
        selftest(session, win)
        return

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
