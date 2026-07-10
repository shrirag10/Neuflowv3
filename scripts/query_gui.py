"""Interactive flow-query GUI (PyQt5) — click anywhere on the image to query
optical flow at that exact point.

The backbone runs ONCE per image pair (~33 ms); every click is a single
decode_queries() call (~1.6 ms), so interaction is effectively instant.
This is the two-pass API doing exactly what it was designed for.

Usage:
    python3 scripts/query_gui.py \
        --img1 datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00100.jpg \
        --img2 datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00101.jpg \
        --checkpoint checkpoints/neuflowv3_chairs_v2dev/step_030000.pth

Controls:
    left click   query flow at the clicked pixel (arrow + value overlay)
    right click  clear all queries
    G            toggle a 32x32 uniform query grid
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils
from utils.load_model import my_load_weights, load_with_new_keys


class FlowSession:
    """One coarse pass, unlimited cheap queries."""

    def __init__(self, img1_path, img2_path, checkpoint, head='convex', pe=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuFlow(use_implicit=True, head_mode=head, use_pe=pe).to(self.device)
        load_with_new_keys(self.model, my_load_weights(checkpoint),
                           missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                           unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
        self.model.eval()

        self.img1_np = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img1 = torch.from_numpy(self.img1_np).permute(2, 0, 1).float()[None]
        img2_np = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        img2 = torch.from_numpy(img2_np).permute(2, 0, 1).float()[None]

        self.padder = frame_utils.InputPadder(img1.shape, padding_factor=16)
        a, b = self.padder.pad(img1.to(self.device), img2.to(self.device))
        self.model.init_bhwd(1, a.shape[-2], a.shape[-1], self.device)

        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            self.state = self.model.infer_coarse_state(a, b)
        self.coarse_ms = (time.perf_counter() - t0) * 1000

    def query(self, points_xy):
        """points_xy: list of (x, y) pixel coords -> (flow [N,2], decode ms)."""
        q = torch.tensor(points_xy, dtype=torch.float32, device=self.device)[None]
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            flow = self.model.decode_queries(self.state, query_coords=q)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return flow[0].float().cpu().numpy(), (time.perf_counter() - t0) * 1000


class QueryWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.queries = []   # [(x, y, u, v), ...]
        self.setWindowTitle('NeuFlow v3 — click to query flow')

        h, w = session.img1_np.shape[:2]
        qimg = QImage(session.img1_np.data, w, h, 3 * w, QImage.Format_RGB888)
        self.base = QPixmap.fromImage(qimg)

        self.label = QLabel()
        self.label.setPixmap(self.base)
        self.label.mousePressEvent = self.on_click
        self.setCentralWidget(self.label)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            f'coarse pass: {session.coarse_ms:.0f} ms (once) — click to query · right-click clears · G = grid')
        self.resize(w, h + 24)

    def on_click(self, ev):
        if ev.button() == Qt.RightButton:
            self.queries.clear()
            self.redraw()
            return
        x, y = ev.pos().x(), ev.pos().y()
        flow, ms = self.session.query([(x, y)])
        u, v = flow[0]
        self.queries.append((x, y, float(u), float(v)))
        self.redraw()
        self.statusBar().showMessage(
            f'({x}, {y}) -> flow ({u:+.2f}, {v:+.2f}) px, |f|={np.hypot(u, v):.2f} px — decoded in {ms:.1f} ms')

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_G:
            h, w = self.session.img1_np.shape[:2]
            xs = np.linspace(8, w - 8, 32)
            ys = np.linspace(8, h - 8, 32)
            pts = [(float(x), float(y)) for y in ys for x in xs]
            flow, ms = self.session.query(pts)
            self.queries = [(p[0], p[1], float(f[0]), float(f[1])) for p, f in zip(pts, flow)]
            self.redraw()
            self.statusBar().showMessage(f'{len(pts)} grid queries decoded in {ms:.1f} ms')

    def redraw(self):
        pm = QPixmap(self.base)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont('Sans', 8)
        painter.setFont(font)
        for x, y, u, v in self.queries:
            mag = np.hypot(u, v)
            hue = int(np.clip(240 - mag * 6, 0, 240))
            color = QColor.fromHsv(hue, 255, 255)
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPoint(int(x), int(y)), QPoint(int(x + u), int(y + v)))
            painter.drawEllipse(QPoint(int(x), int(y)), 3, 3)
            if len(self.queries) <= 20:
                painter.drawText(int(x) + 6, int(y) - 6, f'({u:+.1f}, {v:+.1f})')
        painter.end()
        self.label.setPixmap(pm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img1', required=True)
    ap.add_argument('--img2', required=True)
    ap.add_argument('--checkpoint', default='checkpoints/neuflowv3_chairs_v2dev/step_030000.pth')
    ap.add_argument('--head', default='convex', choices=['regress', 'convex'])
    ap.add_argument('--pe', action='store_true')
    ap.add_argument('--selftest', action='store_true', help='offscreen render + synthetic clicks, save PNG')
    args = ap.parse_args()

    if args.selftest:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    app = QApplication(sys.argv)
    session = FlowSession(args.img1, args.img2, args.checkpoint, args.head, args.pe)
    win = QueryWindow(session)

    if args.selftest:
        h, w = session.img1_np.shape[:2]
        pts = [(w // 4, h // 2), (w // 2, h // 3), (3 * w // 4, 2 * h // 3), (w // 2, 3 * h // 4)]
        flow, ms = session.query(pts)
        win.queries = [(p[0], p[1], float(f[0]), float(f[1])) for p, f in zip(pts, flow)]
        win.redraw()
        win.label.pixmap().save('results/visuals/query_gui_selftest.png')
        print(f'selftest: coarse {session.coarse_ms:.0f} ms, {len(pts)} clicks in {ms:.1f} ms')
        for p, f in zip(pts, flow):
            print(f'  ({p[0]:4d},{p[1]:4d}) -> ({f[0]:+7.2f}, {f[1]:+7.2f}) px')
        return

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
