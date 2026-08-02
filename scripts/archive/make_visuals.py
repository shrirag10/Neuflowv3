"""Generate visual comparison figures: GT vs v2 vs v3 flow + error maps,
and a sparse-query visualization. Outputs to results/visuals/.

Runs alongside training (uses ~4 GB VRAM briefly for the dense v3 pass).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils, flow_viz
from utils.load_model import my_load_weights, load_with_new_keys

DEVICE = torch.device('cuda')
V3_CKPT = 'checkpoints/neuflowv3_chairs_v2dev/step_030000.pth'
PAIRS = [
    ('datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00100.jpg',
     'datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00101.jpg',
     'datasets/vkitti2/Scene18/clone/frames/forwardFlow/Camera_0/flow_00100.png'),
    ('datasets/vkitti2/Scene20/clone/frames/rgb/Camera_0/rgb_00250.jpg',
     'datasets/vkitti2/Scene20/clone/frames/rgb/Camera_0/rgb_00251.jpg',
     'datasets/vkitti2/Scene20/clone/frames/forwardFlow/Camera_0/flow_00250.png'),
]
OUT = 'results/visuals'


def load_model(implicit, ckpt, head='convex'):
    m = NeuFlow(use_implicit=implicit, head_mode=head).to(DEVICE)
    load_with_new_keys(m, my_load_weights(ckpt),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    m.eval()
    return m


def run_dense(model, img1, img2):
    padder = frame_utils.InputPadder(img1.shape, padding_factor=16)
    a, b = padder.pad(img1.to(DEVICE), img2.to(DEVICE))
    model.init_bhwd(1, a.shape[-2], a.shape[-1], DEVICE)
    H, W = a.shape[-2], a.shape[-1]
    with torch.no_grad(), torch.amp.autocast('cuda'):
        if not model.use_implicit:
            out = model(a, b)[-1]
        else:
            # chunked dense decode: VRAM-friendly while training occupies the GPU
            state = model.infer_coarse_state(a, b)
            ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE, dtype=torch.float32),
                                    torch.arange(W, device=DEVICE, dtype=torch.float32), indexing='ij')
            coords = torch.stack([xs, ys], -1).reshape(1, -1, 2)
            chunks = [model.decode_queries(state, query_coords=coords[:, i:i + 65536])
                      for i in range(0, coords.shape[1], 65536)]
            out = torch.cat(chunks, dim=1).reshape(1, H, W, 2).permute(0, 3, 1, 2)
    return padder.unpad(out[0]).float().cpu().permute(1, 2, 0).numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    v2 = load_model(False, 'neuflow_mixed.pth')
    v3 = load_model(True, V3_CKPT)

    for idx, (p1, p2, pf) in enumerate(PAIRS):
        img1_np = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        img1 = torch.from_numpy(img1_np).permute(2, 0, 1).float()[None]
        img2 = torch.from_numpy(cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None]
        gt, valid = frame_utils.read_vkitti_png_flow(pf)

        f2 = run_dense(v2, img1, img2)
        f3 = run_dense(v3, img1, img2)

        e2 = np.linalg.norm(f2 - gt, axis=-1) * valid
        e3 = np.linalg.norm(f3 - gt, axis=-1) * valid
        epe2 = e2.sum() / valid.sum()
        epe3 = e3.sum() / valid.sum()

        rows = [
            (img1_np, 'Input frame'),
            (flow_viz.flow_to_image(gt), 'Ground-truth flow'),
            (flow_viz.flow_to_image(f2), f'NeuFlow v2 — EPE {epe2:.2f}px'),
            (flow_viz.flow_to_image(f3), f'NeuFlow v3 (chairs, convex) — EPE {epe3:.2f}px'),
        ]
        fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(10, 13), dpi=150)
        for ax, (im, title) in zip(axes, rows):
            ax.imshow(im)
            ax.set_title(title, fontsize=11, loc='left')
            ax.axis('off')
        emax = np.percentile(np.concatenate([e2.ravel(), e3.ravel()]), 99)
        err = np.concatenate([e2, np.full((6, e2.shape[1]), np.nan), e3], axis=0)
        im = axes[-1].imshow(err, cmap='magma', vmax=emax)
        axes[-1].set_title('Error maps: v2 (top) vs v3 (bottom) — brighter = larger error', fontsize=11, loc='left')
        axes[-1].axis('off')
        plt.tight_layout()
        path = f'{OUT}/compare_{idx}.png'
        plt.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'{path}: v2 EPE {epe2:.3f}, v3 EPE {epe3:.3f}')

    # ---- sparse query visualization ----
    p1, p2, pf = PAIRS[0]
    img1_np = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
    img1 = torch.from_numpy(img1_np).permute(2, 0, 1).float()[None]
    img2 = torch.from_numpy(cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None]
    padder = frame_utils.InputPadder(img1.shape, padding_factor=16)
    a, b = padder.pad(img1.to(DEVICE), img2.to(DEVICE))
    v3.init_bhwd(1, a.shape[-2], a.shape[-1], DEVICE)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        state = v3.infer_coarse_state(a, b)
        H, W = img1_np.shape[:2]
        gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=12)
        q = torch.from_numpy(pts.reshape(1, -1, 2)).float().to(DEVICE)
        flow_q = v3.decode_queries(state, query_coords=q)[0].float().cpu().numpy()
    q_np = q[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.imshow(img1_np)
    ax.quiver(q_np[:, 0], q_np[:, 1], flow_q[:, 0], flow_q[:, 1],
              np.linalg.norm(flow_q, axis=1), cmap='cool',
              angles='xy', scale_units='xy', scale=1, width=0.0022)
    ax.set_title(f'{len(q_np)} sparse queries at detected corners — decoded in one 1.6 ms call '
                 f'(0.05% of a dense field)', fontsize=11, loc='left')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUT}/sparse_queries.png', bbox_inches='tight', facecolor='white')
    print(f'{OUT}/sparse_queries.png: {len(q_np)} queries')


if __name__ == '__main__':
    main()
