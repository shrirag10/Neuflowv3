"""NeuFlow v3 — interactive flow query tool (PyQt5).

The backbone runs once per image pair (~33 ms); every interaction afterwards
is a decode_queries() call (~1.6 ms). Demonstrates the two-pass O(N) API.

Usage:
    python3 scripts/query_gui.py [--img1 A.jpg --img2 B.jpg] [--checkpoint C.pth]
                                 [--head convex] [--pe] [--selftest]

Interactions:
    left click     query flow at that pixel (arrow + value)
    right click    clear all queries
    G              decode a uniform grid (size set in the toolbar)
    A              adaptive queries at motion boundaries (N set in the toolbar)
    D              toggle dense flow overlay (full-field visualization)
    File menu      open a new image pair / checkpoint, export CSV / screenshot
"""

import sys, os, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QStatusBar, QFileDialog, QMessageBox,
    QAction, QToolBar, QSpinBox, QScrollArea, QWidget, QVBoxLayout,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils, flow_viz
from utils.load_model import my_load_weights, load_with_new_keys

DEFAULT_CKPT = 'checkpoints/neuflowv3_chairs_v2dev/step_030000.pth'


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

    def set_pair(self, img1_path, img2_path):
        self.img1_np = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img1 = torch.from_numpy(self.img1_np).permute(2, 0, 1).float()[None]
        img2_np = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        img2 = torch.from_numpy(img2_np).permute(2, 0, 1).float()[None]
        self.padder = frame_utils.InputPadder(img1.shape, padding_factor=16)
        a, b = self.padder.pad(img1.to(self.device), img2.to(self.device))
        self.pad_hw = (a.shape[-2], a.shape[-1])
        self.model.init_bhwd(1, a.shape[-2], a.shape[-1], self.device)
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            self.state = self.model.infer_coarse_state(a, b)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.coarse_ms = (time.perf_counter() - t0) * 1000

    def query(self, points_xy):
        q = torch.tensor(points_xy, dtype=torch.float32, device=self.device)[None]
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            flow = self.model.decode_queries(self.state, query_coords=q)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return flow[0].float().cpu().numpy(), (time.perf_counter() - t0) * 1000

    def adaptive(self, n):
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            from NeuFlow.adaptive_query import coarse_flow_query
            q = coarse_flow_query(self.state['coarse_flow_s8'], num_points=n, adaptive_ratio=0.7)
            flow = self.model.decode_queries(self.state, query_coords=q)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return q[0].cpu().numpy(), flow[0].float().cpu().numpy(), (time.perf_counter() - t0) * 1000

    def dense(self, chunk=65536):
        H, W = self.pad_hw
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            ys, xs = torch.meshgrid(torch.arange(H, device=self.device, dtype=torch.float32),
                                    torch.arange(W, device=self.device, dtype=torch.float32), indexing='ij')
            coords = torch.stack([xs, ys], -1).reshape(1, -1, 2)
            parts = [self.model.decode_queries(self.state, query_coords=coords[:, i:i + chunk])
                     for i in range(0, coords.shape[1], chunk)]
            out = torch.cat(parts, dim=1).reshape(1, H, W, 2).permute(0, 3, 1, 2)
        out = self.padder.unpad(out[0]).float().cpu().permute(1, 2, 0).numpy()
        return out, (time.perf_counter() - t0) * 1000


class QueryWindow(QMainWindow):
    def __init__(self, session, img1=None, img2=None):
        super().__init__()
        self.session = session
        self.queries = []
        self.dense_overlay = None
        self.show_dense = False
        self.setWindowTitle('NeuFlow v3 — flow query tool')

        self.label = QLabel('File → Open image pair… to begin')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.mousePressEvent = self.on_click
        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        self.setStatusBar(QStatusBar())
        self._build_menu()
        self._build_toolbar()
        self.resize(1280, 500)

        if img1 and img2:
            self.load_pair(img1, img2)

    # ---- UI scaffolding -------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu('&File')
        for text, slot, key in [
            ('Open image pair…', self.open_pair, 'Ctrl+O'),
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
            'Left click: query flow at pixel\nRight click: clear\n'
            'G: uniform grid (toolbar sets size)\nA: adaptive queries at motion boundaries\n'
            'D: toggle dense flow overlay\n\nEach click decodes in ~1.6 ms; the backbone ran once.'))
        h.addAction(a)

    def _build_toolbar(self):
        tb = QToolBar('Query settings')
        self.addToolBar(tb)
        tb.addWidget(QLabel(' Grid: '))
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

    # ---- actions ---------------------------------------------------------
    def open_pair(self):
        p1, _ = QFileDialog.getOpenFileName(self, 'Frame 1', 'datasets', 'Images (*.jpg *.png *.ppm)')
        if not p1:
            return
        p2, _ = QFileDialog.getOpenFileName(self, 'Frame 2', os.path.dirname(p1), 'Images (*.jpg *.png *.ppm)')
        if p2:
            self.load_pair(p1, p2)

    def load_pair(self, p1, p2):
        try:
            self.session.set_pair(p1, p2)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Could not process pair:\n{e}')
            return
        h, w = self.session.img1_np.shape[:2]
        qimg = QImage(self.session.img1_np.data, w, h, 3 * w, QImage.Format_RGB888)
        self.base = QPixmap.fromImage(qimg)
        self.queries.clear()
        self.dense_overlay = None
        self.show_dense = False
        self.redraw()
        self.statusBar().showMessage(
            f'{os.path.basename(p1)} → {os.path.basename(p2)} · coarse pass {self.session.coarse_ms:.0f} ms '
            f'(runs once) · {os.path.basename(self.session.checkpoint)} · click to query')

    def open_checkpoint(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Checkpoint', 'checkpoints', 'Checkpoints (*.pth)')
        if p:
            try:
                self.session = FlowSession(p)
                self.statusBar().showMessage(f'Loaded {os.path.basename(p)} — open an image pair')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Could not load checkpoint:\n{e}')

    def on_click(self, ev):
        if self.session.state is None:
            return
        if ev.button() == Qt.RightButton:
            self.clear()
            return
        x, y = ev.pos().x(), ev.pos().y()
        h, w = self.session.img1_np.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        flow, ms = self.session.query([(x, y)])
        u, v = flow[0]
        self.queries.append((x, y, float(u), float(v)))
        self.redraw()
        self.statusBar().showMessage(
            f'query ({x}, {y}) → flow ({u:+.2f}, {v:+.2f}) px · |f| = {np.hypot(u, v):.2f} px · decoded in {ms:.1f} ms')

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
                painter.setPen(QPen(color, 2))

        # magnitude legend (blue = slow, red = fast)
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
    h, w = session.img1_np.shape[:2]
    flow, ms1 = session.query([(w // 4, h // 2), (3 * w // 4, h // 2)])
    win.queries = [(w // 4, h // 2, float(flow[0][0]), float(flow[0][1])),
                   (3 * w // 4, h // 2, float(flow[1][0]), float(flow[1][1]))]
    win.grid_spin.setValue(24)
    win.run_grid()
    win.run_adaptive()
    win.toggle_dense()
    win.redraw()
    os.makedirs(outdir, exist_ok=True)
    win.label.pixmap().save(f'{outdir}/query_gui_selftest.png')
    with open('/tmp/gui_queries.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        for q in win.queries[:5]:
            wr.writerow(q)
    print(f'selftest OK: coarse {session.coarse_ms:.0f} ms · 2 clicks {ms1:.1f} ms · '
          f'grid+adaptive+dense exercised · screenshot saved')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img1'); ap.add_argument('--img2')
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
    win = QueryWindow(session, args.img1, args.img2)

    if args.selftest:
        selftest(session, win)
        return

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
