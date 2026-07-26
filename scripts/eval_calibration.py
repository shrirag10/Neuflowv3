"""Check whether the uncertainty head's predicted error scale b actually
correlates with real per-point error. Samples N query points per image
(mix of uniform + boundary-biased, matching training distribution), compares
predicted b against true |pred - GT|.

A calibrated head: high b <-> high real error (positive correlation), and
points sorted into low/high-b bins should show a monotonic real-error trend.
This does NOT re-check flow accuracy (already measured in eval_vkitti2.py) —
only whether the confidence signal is meaningful.

Usage:
    python3 scripts/eval_calibration.py --checkpoint <uncG checkpoint>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from tqdm import tqdm

from NeuFlow.neuflow import NeuFlow
from utils.load_model import my_load_weights, load_with_new_keys
from scripts.eval_vkitti2 import build_vkitti2_val_pairs, read_vkitti2_flow
from data_utils import frame_utils


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--val_scenes', nargs='+', default=['Scene18', 'Scene20'])
    ap.add_argument('--n_per_image', type=int, default=2000)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'

    model = NeuFlow(use_implicit=True, head_mode='convex', predict_uncertainty=True).to(device)
    load_with_new_keys(model, my_load_weights(args.checkpoint),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    model.eval()

    pairs = build_vkitti2_val_pairs(args.dataset_root, args.val_scenes)
    print(f'Val pairs: {len(pairs)}')

    import cv2
    all_b, all_err = [], []

    for p1, p2, pf in tqdm(pairs):
        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None]
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float()[None]
        flow_gt, valid = read_vkitti2_flow(pf)

        padder = frame_utils.InputPadder(t1.shape, padding_factor=16)
        t1, t2 = padder.pad(t1.to(device), t2.to(device))
        model.init_bhwd(1, t1.shape[-2], t1.shape[-1], device, amp=amp)

        valid_yx = valid.bool().nonzero(as_tuple=False)
        if valid_yx.shape[0] == 0:
            continue
        n = min(args.n_per_image, valid_yx.shape[0])
        idx = torch.randperm(valid_yx.shape[0])[:n]
        sel = valid_yx[idx]
        # InputPadder (mode='sintel', the default) pads symmetrically on all
        # sides, so raw (y, x) from the unpadded GT must be shifted by
        # (pad_top, pad_left) to index the same content in the padded image.
        pad_left, pad_top = padder._pad[0], padder._pad[2]
        q = torch.stack([sel[:, 1].float() + pad_left, sel[:, 0].float() + pad_top],
                        dim=-1)[None].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            state = model.infer_coarse_state(t1, t2)
            flow, b = model.decode_queries(state, query_coords=q, return_uncertainty=True)

        flow = flow[0].float().cpu().numpy()
        b_np = b[0].float().cpu().numpy()
        gt_at_q = flow_gt[:, sel[:, 0], sel[:, 1]].numpy().T  # [n, 2]
        err = np.linalg.norm(flow - gt_at_q, axis=-1)

        all_b.append(b_np)
        all_err.append(err)

    b_all = np.concatenate(all_b)
    err_all = np.concatenate(all_err)

    corr = np.corrcoef(b_all, err_all)[0, 1]
    print(f'\nSamples: {len(b_all):,}')
    print(f'Predicted b:  mean {b_all.mean():.3f}  median {np.median(b_all):.3f}  '
          f'[{b_all.min():.3f}, {b_all.max():.3f}]')
    print(f'Real error:   mean {err_all.mean():.3f}  median {np.median(err_all):.3f}')
    print(f'Pearson correlation (b vs real error): {corr:.4f}')

    print('\nBinned by predicted b (should show monotonically increasing real error):')
    order = np.argsort(b_all)
    bins = np.array_split(order, 5)
    for i, idxs in enumerate(bins):
        lo, hi = b_all[idxs].min(), b_all[idxs].max()
        print(f'  bin {i+1} (b in [{lo:.2f}, {hi:.2f}]): mean real error = {err_all[idxs].mean():.3f} px')

    print('\nVerdict:', 'CALIBRATED (positive correlation, monotonic bins expected)'
          if corr > 0.15 else 'NOT CLEARLY CALIBRATED (weak/no correlation)')


if __name__ == '__main__':
    main()
