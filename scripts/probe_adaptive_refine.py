"""Can refinement effort be spent only where it changes the answer?

The s8 refinement loop is 59% of NeuFlow's runtime (8 iters x 2.55 ms of
34.4 ms). If iterations split+1..8 can be skipped wherever the flow has already
converged, that stage shrinks proportionally.

This measures, on FULL frames over many pairs (an early version used a small
corner crop and produced unusable numbers -- the crop was easy content where
refinement does nothing, and the sign of the effect flipped between samples):

  1. how concentrated the iteration split->8 change is,
  2. whether it is predictable from signals available AT iteration split,
  3. the EPE when only the top-X% of tiles receive the later iterations,
  4. the halo-adjusted compute actually saved.

Reports absolute EPE and per-pair spread. Percentages of "gain recovered" are
only printed when the available gain is large enough to be meaningful.

Usage (GPU, full val set):
  python3 scripts/probe_adaptive_refine.py --pairs 100 --tile 16
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

from NeuFlow.neuflow import NeuFlow
from NeuFlow import config
from utils.load_model import my_load_weights, load_with_new_keys
from data_utils import frame_utils

# refine_s8 is 8 stacked 3x3 convs -> receptive field radius 8 s8-cells per
# iteration, so a gathered tile needs an 8-cell halo on every side.
HALO = 8


@torch.no_grad()
def coarse_iters(model, img0, img1, iters_s16=1, iters_s8=8):
    """infer_coarse_state, but returns the flow after every s8 iteration.
    Verified to reproduce infer_coarse_state exactly (max diff 0.0)."""
    img0 = img0 / 255.
    img1 = img1 / 255.
    f16, f8 = model.backbone(torch.cat([img0, img1], dim=0))
    f16 = model.cross_attn_s16(f16)
    f16, c16 = model.split_features(f16, config.context_dim_s16, config.feature_dim_s16)
    f8, c8 = model.split_features(f8, config.context_dim_s8, config.feature_dim_s8)
    f0_16, f1_16 = f16.chunk(2, dim=0)

    flow = model.matching_s16.global_correlation_softmax(f0_16, f1_16)
    pyr16 = model.corr_block_s16.init_corr_pyr(f0_16, f1_16)
    ic16 = model.init_iter_context_s16
    for _ in range(iters_s16):
        corrs = model.corr_block_s16(pyr16, flow)
        ic16, d = model.refine_s16(corrs, c16, ic16, flow)
        flow = flow + d

    flow = F.interpolate(flow, scale_factor=2, mode='nearest') * 2
    f16u = F.interpolate(f16, scale_factor=2, mode='nearest')
    f8m = model.merge_s8(torch.cat([f8, f16u], dim=1))
    f0_8, f1_8 = f8m.chunk(2, dim=0)
    pyr8 = model.corr_block_s8.init_corr_pyr(f0_8, f1_8)
    c16u = F.interpolate(c16, scale_factor=2, mode='nearest')
    c8m = model.context_merge_s8(torch.cat([c8, c16u], dim=1))
    ic8 = model.init_iter_context_s8

    hist = []
    for _ in range(iters_s8):
        corrs = model.corr_block_s8(pyr8, flow)
        ic8, d = model.refine_s8(corrs, c8m, ic8, flow)
        flow = flow + d
        hist.append(flow.clone().float())
    return hist


def val_pairs(root, scenes=('Scene18', 'Scene20')):
    out = []
    for sc in scenes:
        d = f'{root}/{sc}/clone/frames'
        import glob as g
        imgs = sorted(g.glob(f'{d}/rgb/Camera_0/*.jpg'))
        fls = sorted(g.glob(f'{d}/forwardFlow/Camera_0/*.png'))
        for i in range(len(fls)):
            if i + 1 < len(imgs):
                out.append((imgs[i], imgs[i + 1], fls[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--root', default='datasets/vkitti2')
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--split', type=int, default=4)
    ap.add_argument('--tile', type=int, default=16, help='tile size in s8 cells')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    model = NeuFlow(use_implicit=True, head_mode='convex').to(dev)
    load_with_new_keys(model, my_load_weights(args.checkpoint),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    model.eval()

    allp = val_pairs(args.root)
    sel = allp[:: max(1, len(allp) // args.pairs)][:args.pairs]
    print(f'{len(sel)} full frames from Scene18+20, tile={args.tile}, split at iter {args.split}\n')

    fracs = [0.05, 0.10, 0.20, 0.30, 0.50]
    acc = {k: [] for k in ['base_split', 'base_full'] + [f'top{f}' for f in fracs] + [f'orc{f}' for f in fracs]}
    conc, corrs = [], []
    T = args.tile

    for p1, p2, pf in tqdm(sel):
        i1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        i2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        gt_np, valid_np = frame_utils.read_vkitti_png_flow(pf)
        gt = torch.from_numpy(gt_np).permute(2, 0, 1)[None].float()
        vmask = torch.from_numpy(valid_np).bool().flatten()

        t1 = torch.from_numpy(i1).permute(2, 0, 1).float()[None]
        t2 = torch.from_numpy(i2).permute(2, 0, 1).float()[None]
        padder = frame_utils.InputPadder(t1.shape, padding_factor=16)
        a, b = padder.pad(t1.to(dev), t2.to(dev))
        model.init_bhwd(1, a.shape[-2], a.shape[-1], dev, amp=amp)
        with torch.amp.autocast(device_type=dev.type, enabled=amp):
            hist = coarse_iters(model, a, b)

        k = args.split - 1
        f_s, f_f = hist[k].float(), hist[-1].float()
        change = torch.norm(f_f - f_s, dim=1)[0] * 8
        conc.append(change.flatten().cpu())

        not_conv = torch.norm(hist[k] - hist[k - 1], dim=1)[0].float() * 8
        mag = torch.norm(f_s, dim=1)[0] * 8
        gy, gx = torch.gradient(mag)
        grad = gy.abs() + gx.abs()
        tm = lambda x: F.avg_pool2d(x[None, None], T, T, ceil_mode=True)[0, 0]
        tc, tn, tg, tmg = tm(change), tm(not_conv), tm(grad), tm(mag)
        cc = lambda x: float(np.corrcoef(x.flatten().cpu().numpy(),
                                         tc.flatten().cpu().numpy())[0, 1])
        corrs.append((cc(tn), cc(tmg), cc(tg)))

        def epe_of(fl):
            up = F.interpolate(fl, scale_factor=8, mode='bilinear', align_corners=False) * 8
            up = padder.unpad(up[0]).cpu()
            return torch.norm(up - gt[0], dim=0).flatten()[vmask].mean().item()

        acc['base_split'].append(epe_of(f_s))
        acc['base_full'].append(epe_of(f_f))
        for fr in fracs:
            kk = max(1, int(round(fr * tn.numel())))
            for tag, score in [('top', tn), ('orc', tc)]:   # practical vs oracle selector
                keep = torch.zeros(score.numel(), device=score.device)
                keep[torch.topk(score.flatten(), kk).indices] = 1
                m = F.interpolate(keep.view_as(score)[None, None].float(),
                                  size=f_s.shape[-2:], mode='nearest')
                acc[f'{tag}{fr}'].append(epe_of(m * f_f + (1 - m) * f_s))

    allc = torch.cat(conc)
    s = torch.sort(allc, descending=True).values
    print(f'\n=== Concentration of the iter{args.split}->8 change ({len(sel)} frames) ===')
    for p in [0.05, 0.10, 0.20, 0.50]:
        print(f'  top {p*100:4.0f}% of pixels carry {s[:int(p*len(s))].sum()/s.sum()*100:5.1f}% of it')
    print(f'  mean {allc.mean():.3f} px | 99th pct {torch.quantile(s, 0.99).item():.3f} px')

    c = np.array(corrs)
    print(f'\n=== Predicting WHERE, from iteration-{args.split} information only ===')
    for nm, i in [('|flow_k - flow_k-1| (not converged)', 0), ('flow magnitude', 1), ('flow gradient', 2)]:
        print(f'  {nm:36s} r = {c[:,i].mean():.3f} +/- {c[:,i].std():.3f}')

    b4 = np.array(acc['base_split']); b8 = np.array(acc['base_full'])
    gain = b4.mean() - b8.mean()
    print(f'\n=== EPE, mean +/- std over {len(sel)} frames ===')
    print(f'  stop at iteration {args.split:<11d} {b4.mean():.4f} +/- {b4.std():.4f}')
    print(f'  full 8 iterations           {b8.mean():.4f} +/- {b8.std():.4f}')
    print(f'  --> gain from iters {args.split+1}-8      {gain:+.4f} px')
    if gain < 0.01:
        print('  WARNING: gain is tiny; the budget table below is not meaningful.')

    cost = lambda f: f * ((T + 2 * HALO) ** 2) / (T ** 2)
    print(f'\n=== Refining only the top-X% of {T}x{T} tiles ===')
    print(f'  {"budget":>8} {"practical":>11} {"oracle":>10} {"kept":>7} {"iters5-8 cost":>14} {"total speedup":>14}')
    for fr in fracs:
        v = np.mean(acc[f'top{fr}']); o = np.mean(acc[f'orc{fr}'])
        rec = (b4.mean() - v) / gain * 100 if gain > 0.01 else float('nan')
        cf = cost(fr)
        saved_ms = (1 - min(cf, 1.0)) * (8 - args.split) * 2.55
        print(f'  {fr*100:6.0f}% {v:11.4f} {o:10.4f} {rec:6.1f}% {cf*100:13.0f}% '
              f'{saved_ms/34.4*100:13.1f}%')
    print(f'\n  (cost includes the {HALO}-cell halo refine_s8 needs: a {T}x{T} tile '
          f'actually computes {T+2*HALO}x{T+2*HALO})')
    print(f'  break-even tile fraction at T={T}: {T**2/(T+2*HALO)**2*100:.0f}%')


if __name__ == '__main__':
    main()
