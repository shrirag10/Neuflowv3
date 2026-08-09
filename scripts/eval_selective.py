"""Risk-coverage for v3 AND the baseline on the same accepted pixels.

The selective-prediction figure in the deck plots the baseline as a flat line at
its full-frame mean, because it has no confidence signal of its own and so has
one operating point. That is true, but it makes the comparison at 80% coverage
read as if the baseline scored 2.324 on the pixels v3 kept, which nobody has
measured. The accepted 80% is the easy 80%, and the baseline is more accurate
there too.

This measures both models at the same query points, ranks by v3's predicted
confidence, and reports the accepted-set error for each at every coverage level.
The gap that survives is the part attributable to the confidence head rather
than to pixel difficulty.

    python3 scripts/eval_selective.py \
        --checkpoint checkpoints/scratch/v3_FlyingChairs_VKITTI2_Sintel_uncertainty/step_100000.pth
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
from tqdm import tqdm

from NeuFlow.neuflow import NeuFlow
from utils.load_model import my_load_weights, load_with_new_keys
from eval_vkitti2 import build_vkitti2_val_pairs, read_vkitti2_flow
from data_utils import frame_utils


def build(ckpt, implicit, dev, uncertainty=False):
    m = NeuFlow(use_implicit=implicit, head_mode='convex',
                predict_uncertainty=uncertainty).to(dev)
    load_with_new_keys(m, my_load_weights(ckpt),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    m.eval()
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='v3 checkpoint WITH the uncertainty head')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--val_scenes', nargs='+', default=['Scene18', 'Scene20'])
    ap.add_argument('--n_per_image', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    torch.manual_seed(args.seed)

    v3 = build(args.checkpoint, True, dev, uncertainty=True)
    v2 = build(args.v2_checkpoint, False, dev)

    pairs = build_vkitti2_val_pairs(args.dataset_root, args.val_scenes)
    print(f'Val pairs: {len(pairs)}')

    import cv2
    B, E3, E2 = [], [], []

    for p1, p2, pf in tqdm(pairs):
        i1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        i2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(i1).permute(2, 0, 1).float()[None]
        t2 = torch.from_numpy(i2).permute(2, 0, 1).float()[None]
        flow_gt, valid = read_vkitti2_flow(pf)

        padder = frame_utils.InputPadder(t1.shape, padding_factor=16)
        pa, pb = padder.pad(t1.to(dev), t2.to(dev))

        valid_yx = valid.bool().nonzero(as_tuple=False)
        if valid_yx.shape[0] == 0:
            continue
        n = min(args.n_per_image, valid_yx.shape[0])
        sel = valid_yx[torch.randperm(valid_yx.shape[0])[:n]]
        ys, xs = sel[:, 0], sel[:, 1]

        # v3: query at exactly those points, in padded-frame coordinates
        pad_left, pad_top = padder._pad[0], padder._pad[2]
        q = torch.stack([xs.float() + pad_left, ys.float() + pad_top], -1)[None].to(dev)
        v3.init_bhwd(1, pa.shape[-2], pa.shape[-1], dev, amp=amp)
        with torch.amp.autocast(device_type=dev.type, enabled=amp):
            st = v3.infer_coarse_state(pa, pb)
            f3, b = v3.decode_queries(st, query_coords=q, return_uncertainty=True)

        # v2: dense forward, unpad, then read the SAME pixels
        v2.init_bhwd(1, pa.shape[-2], pa.shape[-1], dev, amp=amp)
        with torch.amp.autocast(device_type=dev.type, enabled=amp):
            f2_dense = v2(pa, pb)[-1]
        f2_dense = padder.unpad(f2_dense[0]).float().cpu()

        gt = flow_gt[:, ys, xs].numpy().T
        e3 = np.linalg.norm(f3[0].float().cpu().numpy() - gt, axis=-1)
        e2 = np.linalg.norm(f2_dense[:, ys, xs].numpy().T - gt, axis=-1)

        B.append(b[0].float().cpu().numpy()); E3.append(e3); E2.append(e2)

    b_all = np.concatenate(B); e3 = np.concatenate(E3); e2 = np.concatenate(E2)
    order = np.argsort(b_all)          # most confident first
    e3s, e2s = e3[order], e2[order]

    print(f'\nQueries: {len(b_all):,}  ({len(pairs)} pairs, {args.n_per_image}/image)')
    print(f'\n{"coverage":>9} {"v3":>8} {"v2 same px":>11} {"gap":>8} {"ratio":>7}')
    print('-' * 48)
    for cov in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = int(len(e3s) * cov)
        a, c = e3s[:k].mean(), e2s[:k].mean()
        print(f'{cov*100:8.0f}% {a:8.3f} {c:11.3f} {a-c:+8.3f} {c/a:7.2f}x')

    print(f'\nFull-frame reference, all sampled points:')
    print(f'  v3 {e3.mean():.4f}   v2 {e2.mean():.4f}')
    print('\nThe deck plots v2 flat at its full-frame mean. The "v2 same px" column is')
    print('what v2 actually scores on the pixels v3 chose to keep.')


if __name__ == '__main__':
    main()
