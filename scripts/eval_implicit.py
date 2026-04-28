import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Evaluate the trained implicit flow decoder at multiple resolutions.

Uses chunked inference for high-resolution to avoid OOM on 8 GB GPUs.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from NeuFlow.neuflow import NeuFlow
from data_utils.frame_utils import read_vkitti_png_flow


# ── Flow → color visualization (Middlebury encoding) ────────────────────────
def flow_to_color(flow, max_flow=None):
    """Convert [H, W, 2] flow to [H, W, 3] BGR color image."""
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u ** 2 + v ** 2)
    if max_flow is None:
        max_flow = max(mag.max(), 1e-5)
    norm_mag = mag / max_flow
    angle = np.arctan2(-v, -u) / np.pi
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + 1) / 2 * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (norm_mag.clip(0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def load_model(checkpoint_path, device):
    model = NeuFlow(use_implicit=True).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)['model']
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference_chunked(model, img0, img1, device, target_h=None, target_w=None, chunk_size=65536):
    """Run inference with chunked sparse queries to avoid OOM at high resolutions."""
    B, _, H, W = img0.shape
    model.init_bhwd(B, H, W, device, amp=True)

    # Compute encoder/correlation/refinement once and reuse for all query chunks.
    with torch.no_grad(), torch.amp.autocast('cuda'):
        state = model.infer_coarse_state(img0, img1, iters_s16=4, iters_s8=7)

    if target_h is None:
        target_h = H
    if target_w is None:
        target_w = W

    total_pixels = target_h * target_w

    if total_pixels <= chunk_size:
        # Small enough — one dense decode pass
        with torch.no_grad(), torch.amp.autocast('cuda'):
            flow = model.decode_queries(state, target_h=target_h, target_w=target_w)
        return flow

    # Process in chunks without materializing a full dense coordinate tensor.
    flow = torch.zeros(B, 2, target_h, target_w, device=device)
    flow_flat = flow.permute(0, 2, 3, 1).reshape(B, total_pixels, 2)

    scale_x = W / target_w
    scale_y = H / target_h

    for start in range(0, total_pixels, chunk_size):
        end = min(start + chunk_size, total_pixels)
        idx = torch.arange(start, end, device=device)
        y = idx // target_w
        x = idx % target_w

        if target_h != H or target_w != W:
            y = (y.float() + 0.5) * scale_y - 0.5
            x = (x.float() + 0.5) * scale_x - 0.5
        else:
            y = y.float()
            x = x.float()

        chunk_coords = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, chunk, 2]

        with torch.no_grad(), torch.amp.autocast('cuda'):
            flow_chunk = model.decode_queries(state, query_coords=chunk_coords)
        # decode_queries returns [B, chunk, 2] for sparse coordinates
        flow_flat[:, start:end, :] = flow_chunk

    return flow


def main():
    device = torch.device('cuda')
    ckpt = 'checkpoints/neuflowv3/step_020000.pth'
    out_dir = 'results/neuflowv3_eval'
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading model from {ckpt}...')
    model = load_model(ckpt, device)

    # Use sample pairs + GT flow
    sample_data = [
        {
            'img0': 'datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00100.jpg',
            'img1': 'datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00101.jpg',
            'flow_gt': 'datasets/vkitti2/Scene01/clone/frames/forwardFlow/Camera_0/flow_00100.png',
        },
        {
            'img0': 'datasets/vkitti2/Scene02/clone/frames/rgb/Camera_0/rgb_00050.jpg',
            'img1': 'datasets/vkitti2/Scene02/clone/frames/rgb/Camera_0/rgb_00051.jpg',
            'flow_gt': 'datasets/vkitti2/Scene02/clone/frames/forwardFlow/Camera_0/flow_00050.png',
        },
        {
            'img0': 'datasets/vkitti2/Scene06/clone/frames/rgb/Camera_0/rgb_00030.jpg',
            'img1': 'datasets/vkitti2/Scene06/clone/frames/rgb/Camera_0/rgb_00031.jpg',
            'flow_gt': 'datasets/vkitti2/Scene06/clone/frames/forwardFlow/Camera_0/flow_00030.png',
        },
    ]

    scales = [1, 2, 4]

    for pair_idx, data in enumerate(sample_data):
        print(f'\n=== Pair {pair_idx}: {os.path.basename(data["img0"])} ===')

        img0_np = cv2.imread(data['img0'])
        img1_np = cv2.imread(data['img1'])
        if img0_np is None or img1_np is None:
            print(f'  Skipping — could not read images')
            continue

        H_orig, W_orig = img0_np.shape[:2]

        # Load GT flow
        gt_flow, gt_valid = read_vkitti_png_flow(data['flow_gt'])

        # Pad to multiple of 16
        pad_h = (16 - H_orig % 16) % 16
        pad_w = (16 - W_orig % 16) % 16
        img0_pad = np.pad(img0_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        img1_pad = np.pad(img1_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        H_pad, W_pad = img0_pad.shape[:2]

        # To tensor (BGR→RGB)
        img0_t = torch.from_numpy(img0_pad[..., ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img1_t = torch.from_numpy(img1_pad[..., ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # Shared max_flow for consistent coloring
        gt_mag = np.sqrt((gt_flow ** 2).sum(-1))
        max_flow_vis = max(gt_mag.max(), 1e-5)

        flow_panels = []
        panel_labels = []

        for scale in scales:
            target_h = H_pad * scale
            target_w = W_pad * scale

            print(f'  Running {scale}× ({target_w}×{target_h})...', end=' ', flush=True)

            flow = run_inference_chunked(
                model, img0_t, img1_t, device,
                target_h=target_h if scale > 1 else None,
                target_w=target_w if scale > 1 else None,
                chunk_size=32768,
            )

            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
            flow_np = flow_np[:H_orig * scale, :W_orig * scale]

            mag = np.sqrt((flow_np ** 2).sum(-1))

            # EPE vs GT (resize GT to match scale)
            if scale == 1:
                gt_crop = gt_flow[:H_orig, :W_orig]
                valid_crop = gt_valid[:H_orig, :W_orig]
            else:
                gt_crop = cv2.resize(gt_flow[:H_orig, :W_orig],
                                     (W_orig * scale, H_orig * scale),
                                     interpolation=cv2.INTER_LINEAR) * scale
                valid_crop = cv2.resize(gt_valid[:H_orig, :W_orig].astype(np.float32),
                                         (W_orig * scale, H_orig * scale),
                                         interpolation=cv2.INTER_NEAREST) > 0.5

            diff = np.sqrt(((flow_np - gt_crop) ** 2).sum(-1))
            epe = diff[valid_crop].mean() if valid_crop.sum() > 0 else 0
            print(f'mag: [{mag.min():.1f}, {mag.max():.1f}] | EPE: {epe:.3f}')

            # Visualize with consistent color scale
            flow_color = flow_to_color(flow_np, max_flow=max_flow_vis * scale)
            vis_h = H_orig
            vis_w = int(flow_color.shape[1] * vis_h / flow_color.shape[0])
            flow_panels.append(cv2.resize(flow_color, (vis_w, vis_h)))
            panel_labels.append(f'{scale}x (EPE {epe:.2f})')

        # GT flow panel
        gt_color = flow_to_color(gt_flow[:H_orig, :W_orig], max_flow=max_flow_vis)
        gt_panel = cv2.resize(gt_color, (flow_panels[0].shape[1], H_orig))

        # Input image panel
        img_panel = cv2.resize(img0_np[:H_orig, :W_orig], (flow_panels[0].shape[1], H_orig))

        all_panels = [img_panel, gt_panel] + flow_panels
        labels = ['Input', 'GT Flow'] + panel_labels

        labeled = []
        for panel, label in zip(all_panels, labels):
            p = panel.copy()
            cv2.putText(p, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(p, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 1, cv2.LINE_AA)
            labeled.append(p)

        composite = np.concatenate(labeled, axis=1)
        save_path = os.path.join(out_dir, f'pair_{pair_idx}_multi_res.png')
        cv2.imwrite(save_path, composite)
        print(f'  Saved: {save_path}')

    print(f'\n✓ All results saved to {out_dir}/')


if __name__ == '__main__':
    main()
