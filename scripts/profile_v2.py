"""Per-stage latency profile of NeuFlow v2 + no-retrain iteration ablation.

Part 1: CUDA-event timing of every pipeline stage at 384x1248 fp16.
Part 2: EPE/latency sweep over refinement iteration counts on a VKITTI2 subset
        (frozen weights — measures how much accuracy each iteration actually buys).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from NeuFlow.neuflow import NeuFlow
from NeuFlow import config
from data_utils import frame_utils
from utils.load_model import my_load_weights, load_with_new_keys

DEVICE = torch.device('cuda')
H, W = 384, 1248


def load_v2():
    m = NeuFlow(use_implicit=False).to(DEVICE)
    load_with_new_keys(m, my_load_weights('neuflow_mixed.pth'),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=[])
    m.eval()
    m.init_bhwd(1, H, W, DEVICE)
    return m


def cuda_ms(fn, warmup=10, runs=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


@torch.no_grad()
def profile_stages():
    m = load_v2()
    img0 = torch.randint(0, 255, (1, 3, H, W), dtype=torch.float32, device=DEVICE) / 255.
    img1 = torch.randint(0, 255, (1, 3, H, W), dtype=torch.float32, device=DEVICE) / 255.

    with torch.amp.autocast('cuda'):
        # precompute intermediates once so each stage can be timed in isolation
        imgs = torch.cat([img0, img1], dim=0)
        f16, f8 = m.backbone(imgs)
        f16a = m.cross_attn_s16(f16)
        feats16, ctx16 = m.split_features(f16a, config.context_dim_s16, config.feature_dim_s16)
        feats8, ctx8 = m.split_features(f8, config.context_dim_s8, config.feature_dim_s8)
        f0_16, f1_16 = feats16.chunk(2, dim=0)
        flow0 = m.matching_s16.global_correlation_softmax(f0_16, f1_16)
        pyr16 = m.corr_block_s16.init_corr_pyr(f0_16, f1_16)
        ic16 = m.init_iter_context_s16

        def refine16_step():
            corrs = m.corr_block_s16(pyr16, flow0)
            m.refine_s16(corrs, ctx16, ic16, flow0)

        flow_up = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2
        f16_up = F.interpolate(feats16, scale_factor=2, mode='nearest')
        feats8m = m.merge_s8(torch.cat([feats8, f16_up], dim=1))
        f0_8, f1_8 = feats8m.chunk(2, dim=0)
        pyr8 = m.corr_block_s8.init_corr_pyr(f0_8, f1_8)
        ctx16_up = F.interpolate(ctx16, scale_factor=2, mode='nearest')
        ctx8m = m.context_merge_s8(torch.cat([ctx8, ctx16_up], dim=1))
        ic8 = m.init_iter_context_s8

        def refine8_step():
            corrs = m.corr_block_s8(pyr8, flow_up)
            m.refine_s8(corrs, ctx8m, ic8, flow_up)

        def upsampler():
            f0_s1 = m.conv_s8(img0)
            m.upsample_s8(f0_s1, flow_up)

        stages = [
            ('backbone (both images)', lambda: m.backbone(imgs)),
            ('cross-attention s16', lambda: m.cross_attn_s16(f16)),
            ('global matching s16', lambda: m.matching_s16.global_correlation_softmax(f0_16, f1_16)),
            ('corr pyramid init s16', lambda: m.corr_block_s16.init_corr_pyr(f0_16, f1_16)),
            ('refine s16 (1 iter)', refine16_step),
            ('merge to s8', lambda: m.merge_s8(torch.cat([feats8, f16_up], dim=1))),
            ('corr pyramid init s8', lambda: m.corr_block_s8.init_corr_pyr(f0_8, f1_8)),
            ('refine s8 (per iter)', refine8_step),
            ('convex upsampler', upsampler),
        ]
        print(f'\n== Per-stage latency (fp16, {H}x{W}) ==')
        total_acc = 0.0
        for name, fn in stages:
            ms = cuda_ms(fn)
            mult = 8 if 'per iter' in name else 1
            total_acc += ms * mult
            print(f'{name:28s} {ms:7.2f} ms' + (f'  x8 = {ms*8:6.2f} ms' if mult == 8 else ''))
        full = cuda_ms(lambda: m(img0 * 255, img1 * 255))
        print(f'{"sum of stages (approx)":28s} {total_acc:7.2f} ms')
        print(f'{"full forward (measured)":28s} {full:7.2f} ms')


@torch.no_grad()
def sweep_iters(n_pairs):
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_vkitti2 import build_vkitti2_val_pairs, read_vkitti2_flow
    pairs = build_vkitti2_val_pairs('datasets/vkitti2', ['Scene18', 'Scene20'])[::max(1, 1174 // n_pairs)]
    print(f'\n== Iteration sweep on {len(pairs)} pairs ==')
    m = load_v2()

    configs = [(1, 8), (1, 6), (1, 4), (1, 3), (1, 2), (1, 1), (0, 8), (0, 4), (2, 4), (1, 12)]
    print(f'{"s16":>4} {"s8":>4} {"EPE":>8} {"1px%":>7} {"3px%":>7} {"ms":>7} {"FPS":>6}')
    for it16, it8 in configs:
        epe_sum, px, a1, a3 = 0.0, 0, 0, 0
        times = []
        for p1, p2, pf in tqdm(pairs, leave=False):
            i1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
            i2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
            t1 = torch.from_numpy(i1).permute(2, 0, 1).float()[None]
            t2 = torch.from_numpy(i2).permute(2, 0, 1).float()[None]
            gt, valid = read_vkitti2_flow(pf)
            padder = frame_utils.InputPadder(t1.shape, padding_factor=16)
            a, b = padder.pad(t1.to(DEVICE), t2.to(DEVICE))
            m.init_bhwd(1, a.shape[-2], a.shape[-1], DEVICE)
            with torch.amp.autocast('cuda'):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = m(a, b, iters_s16=it16, iters_s8=it8)[-1]
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            f = padder.unpad(out[0]).float().cpu()
            e = torch.norm(f - gt, dim=0).view(-1)[valid.bool().view(-1)]
            epe_sum += e.sum().item(); px += e.numel()
            a1 += (e < 1).sum().item(); a3 += (e < 3).sum().item()
        ms = np.mean(times[5:]) * 1000
        print(f'{it16:>4} {it8:>4} {epe_sum/px:8.3f} {a1/px*100:7.2f} {a3/px*100:7.2f} {ms:7.1f} {1000/ms:6.1f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=int, default=200)
    ap.add_argument('--skip_stages', action='store_true')
    args = ap.parse_args()
    if not args.skip_stages:
        profile_stages()
    sweep_iters(args.pairs)
