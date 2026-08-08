"""Illustrate the three fast-platform scenarios on real frames.

One row per scenario: the frame with its regions marked, and the flow the
decoder returns inside those regions. Uses the same ROI geometry as
scripts/bench_scenarios.py, so the picture and the numbers describe the same
thing.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data_utils import flow_viz
from eval_vkitti2 import build_vkitti2_val_pairs
from flow_engine import FlowEngine
from bench_scenarios import boxes_for

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif']})
INK, HL, WARN = '#1a1a1a', '#0b6a63', '#b03030'
OUT = 'results/plots'

TITLES = {
    'S1': ('First encounter',
           'something worth flowing enters the field of view'),
    'S2': ('Turn',
           'the tracked region now overlaps a second object'),
    'S3': ('New object',
           'a second region appears in a frame already being processed'),
}


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/hpc/v3_best.pth')
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--roi', type=int, default=192)
    ap.add_argument('--pair', type=int, default=430,
                    help='index into the val pair list; pick one with visible motion')
    args = ap.parse_args()

    pairs = build_vkitti2_val_pairs(args.dataset_root, ['Scene18', 'Scene20'])
    p0, p1, _ = pairs[min(args.pair, len(pairs) - 1)]
    i0 = cv2.cvtColor(cv2.imread(p0), cv2.COLOR_BGR2RGB)
    i1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)

    eng = FlowEngine(args.checkpoint, None)
    H, W = i0.shape[:2]
    eng.warmup(H, W)
    state = eng.coarse(i0, i1, key='fig')

    fig, axes = plt.subplots(3, 2, figsize=(16, 7.2), dpi=140,
                             gridspec_kw={'width_ratios': [1.55, 1]})

    for r, sc in enumerate(['S1', 'S2', 'S3']):
        boxes = boxes_for(sc, i0, args.roi)
        title, sub = TITLES[sc]

        # left: the frame with the regions marked
        ax = axes[r][0]
        ax.imshow(i0)
        for bi, b in enumerate(boxes):
            col = HL if bi == 0 else WARN
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, edgecolor=col, lw=2.4))
            # second label goes to the bottom edge so overlapping boxes in S2
            # do not put the two captions on top of each other
            ly = b[1] + 22 if bi == 0 else b[3] - 8
            lx = b[0] + 6 if bi == 0 else b[2] - 46
            ax.text(lx, ly, 'tracked' if bi == 0 else 'new',
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor=col, edgecolor='none', pad=1.6))
        ax.set_title(f'{sc}   {title}   —   {sub}', fontsize=11.5,
                     loc='left', fontweight='bold', color=INK)
        ax.set_xticks([]); ax.set_yticks([])

        # right: the flow actually returned inside those regions
        ax = axes[r][1]
        canvas = np.full((H, W, 3), 248, dtype=np.uint8)
        for b in boxes:
            f, _, _ = eng.query_region(state, b[0], b[1], b[2] - b[0], b[3] - b[1])
            canvas[b[1]:b[3], b[0]:b[2]] = flow_viz.flow_to_image(f)
        ax.imshow(canvas)
        for bi, b in enumerate(boxes):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, edgecolor=HL if bi == 0 else WARN, lw=1.6))
        ax.set_title('flow returned — computed only inside the regions',
                     fontsize=10.5, loc='left', color=INK)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Three situations a fast platform meets, and what is actually computed in each',
                 fontsize=13, x=0.012, ha='left', fontweight='bold', color=INK)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(f'{OUT}/scenarios_illustrated.png', bbox_inches='tight', facecolor='white')
    print(f'{OUT}/scenarios_illustrated.png')


if __name__ == '__main__':
    main()
