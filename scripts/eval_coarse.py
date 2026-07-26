"""Evaluate the coarse 1/8 flow directly (bilinear x8 upsample), bypassing the
decoder entirely. This is the correct test for refinement changes (e.g. option
A self-distillation): it isolates exactly what refine_s8/refine_s16 produce,
independent of the decoder head (which may be untrained/irrelevant for such
checkpoints, e.g. distill3 was never given a trained decoder).

Usage:
    python3 scripts/eval_coarse.py --checkpoint <path> --iters_s16 1 --iters_s8 3
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
import torch.nn.functional as F
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
    ap.add_argument('--iters_s16', type=int, default=1)
    ap.add_argument('--iters_s8', type=int, default=8)
    ap.add_argument('--padding_factor', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'

    model = NeuFlow(use_implicit=True).to(device)
    load_with_new_keys(model, my_load_weights(args.checkpoint),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    model.eval()

    pairs = build_vkitti2_val_pairs(args.dataset_root, args.val_scenes)
    print(f'Val pairs: {len(pairs)}  iters=({args.iters_s16},{args.iters_s8})')

    import cv2
    px_count, epe_sum, acc1, acc3, times = 0, 0.0, 0, 0, []

    for p1, p2, pf in tqdm(pairs):
        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None]
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float()[None]
        flow_gt, valid = read_vkitti2_flow(pf)

        padder = frame_utils.InputPadder(t1.shape, padding_factor=args.padding_factor)
        t1, t2 = padder.pad(t1.to(device), t2.to(device))
        model.init_bhwd(1, t1.shape[-2], t1.shape[-1], device, amp=amp)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            state = model.infer_coarse_state(t1, t2, iters_s16=args.iters_s16, iters_s8=args.iters_s8)
            coarse = state['coarse_flow_s8']
            flow_full = F.interpolate(coarse.float(), scale_factor=8, mode='bilinear', align_corners=False) * 8
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        flow_pr = padder.unpad(flow_full[0]).cpu()
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        v = valid.bool().view(-1)
        e = epe.view(-1)[v]
        px_count += e.numel()
        epe_sum += e.sum().item()
        acc1 += (e < 1.0).sum().item()
        acc3 += (e < 3.0).sum().item()

    t = np.array(times[10:]) * 1000
    print(f'\nCoarse-flow eval ({args.val_scenes}), iters=({args.iters_s16},{args.iters_s8})')
    print(f'  Mean EPE : {epe_sum / px_count:.4f} px')
    print(f'  1px acc  : {acc1 / px_count * 100:.2f}%')
    print(f'  3px acc  : {acc3 / px_count * 100:.2f}%')
    print(f'  Coarse-pass latency: {t.mean():.1f} ms  ({1000 / t.mean():.1f} FPS)')


if __name__ == '__main__':
    main()
