"""Headless proof capture: run the GUI pipeline on a YouTube video, play N
frame pairs with motion detection, and save screenshots of the viewer and the
system-resources tab as evidence for the deck.

Usage:
    QT_QPA_PLATFORM=offscreen python3 scripts/capture_youtube_proof.py \
        --youtube URL --steps 40
"""

import sys, os, time, argparse
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from query_gui import FlowSession, QueryWindow

OUT = 'results/visuals'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--youtube', default='https://www.youtube.com/watch?v=wqctLW0Hb_0')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--start', type=int, default=300, help='first frame (skip intro)')
    args = ap.parse_args()

    app = QApplication(sys.argv)
    session = FlowSession()
    win = QueryWindow(session, youtube=args.youtube)
    win.resize(1400, 700)
    win.show()

    win.show_video_pair(args.start)
    fps_log = []
    win._t_last = time.perf_counter()
    for k in range(args.steps):
        win.play_step()
        if win.play_fps:
            fps_log.append(win.play_fps)
        win.resources_panel.sample()
        app.processEvents()

    os.makedirs(OUT, exist_ok=True)
    win.label.pixmap().save(f'{OUT}/yt_run_viewer.png')
    win.resources_panel.grab().save(f'{OUT}/yt_run_resources.png')

    import numpy as np
    print(f'ran {len(fps_log)} pairs from frame {args.start}: '
          f'mean {np.mean(fps_log):.1f} FPS, median {np.median(fps_log):.1f}, '
          f'motion boxes on last frame: {len(win.motion_boxes)}')
    print(f'saved {OUT}/yt_run_viewer.png and {OUT}/yt_run_resources.png')


if __name__ == '__main__':
    main()
