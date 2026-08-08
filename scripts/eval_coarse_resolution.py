"""Can the coarse pass be run at reduced resolution and the ROI recovered by
querying at full resolution?

This is the question a fast-moving platform forces: the coarse pass is 87% of
the cost and scales with area, so halving resolution is ~4x cheaper. v2 must
then bilinearly upscale its low-res output. v3 can instead query the decoder at
full-resolution coordinates, which is a different operation and the only reason
to expect it to do better.

Both models are scored against full-resolution ground truth, so the comparison
is like-for-like: what does the consumer receive at full resolution.

    python3 scripts/eval_coarse_resolution.py --checkpoint <ckpt> --limit 120
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils
from utils.load_model import my_load_weights, load_with_new_keys
from eval_vkitti2 import build_vkitti2_val_pairs, read_vkitti2_flow


def make(ckpt, implicit, dev, head='convex'):
    m = NeuFlow(use_implicit=implicit, head_mode=head).to(dev)
    load_with_new_keys(m, my_load_weights(ckpt),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'] if implicit else [])
    m.eval()
    return m


@torch.no_grad()
def v3_at_scale(model, t1, t2, scale, out_h, out_w, amp, timer=None):
    """Coarse pass at `scale`, then query the decoder at full-resolution coords.

    The queries are full-resolution pixel positions expressed in the downscaled
    image's coordinate frame, so the decoder is genuinely being asked for values
    between its own input samples.
    """
    if scale != 1.0:
        a = F.interpolate(t1, scale_factor=scale, mode='bilinear', align_corners=False)
        b = F.interpolate(t2, scale_factor=scale, mode='bilinear', align_corners=False)
    else:
        a, b = t1, t2
    padder = frame_utils.InputPadder(a.shape, padding_factor=16)
    pa, pb = padder.pad(a, b)
    model.init_bhwd(1, pa.shape[-2], pa.shape[-1], t1.device, amp=amp)

    if timer is not None:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.amp.autocast('cuda', enabled=amp):
        st = model.infer_coarse_state(pa, pb)
    if timer is not None:
        torch.cuda.synchronize(); timer['coarse'] = (time.perf_counter() - t0) * 1000

    in_h, in_w = a.shape[-2], a.shape[-1]
    pl, pt = padder._pad[0], padder._pad[2]
    xs = (torch.arange(out_w, device=t1.device, dtype=torch.float32) + 0.5) * (in_w / out_w) - 0.5 + pl
    ys = (torch.arange(out_h, device=t1.device, dtype=torch.float32) + 0.5) * (in_h / out_h) - 0.5 + pt

    if timer is not None:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    out = torch.empty(1, out_h, out_w, 2, device=t1.device)
    for y0 in range(0, out_h, 96):
        y1 = min(out_h, y0 + 96)
        gy, gx = torch.meshgrid(ys[y0:y1], xs, indexing='ij')
        q = torch.stack([gx, gy], -1).reshape(1, -1, 2)
        with torch.amp.autocast('cuda', enabled=amp):
            f = model.decode_queries(st, query_coords=q)
        out[:, y0:y1] = f.reshape(1, y1 - y0, out_w, 2).float()
    if timer is not None:
        torch.cuda.synchronize(); timer['decode'] = (time.perf_counter() - t0) * 1000

    # decoder returns displacement in its own (downscaled) pixel units
    return out[0].permute(2, 0, 1) / scale


@torch.no_grad()
def v2_at_scale(model, t1, t2, scale, out_h, out_w, amp, timer=None):
    """v2's only option at reduced resolution: predict small, upscale after."""
    if scale != 1.0:
        a = F.interpolate(t1, scale_factor=scale, mode='bilinear', align_corners=False)
        b = F.interpolate(t2, scale_factor=scale, mode='bilinear', align_corners=False)
    else:
        a, b = t1, t2
    padder = frame_utils.InputPadder(a.shape, padding_factor=16)
    pa, pb = padder.pad(a, b)
    model.init_bhwd(1, pa.shape[-2], pa.shape[-1], t1.device, amp=amp)
    if timer is not None:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.amp.autocast('cuda', enabled=amp):
        o = model(pa, pb)[-1]
    if timer is not None:
        torch.cuda.synchronize(); timer['coarse'] = (time.perf_counter() - t0) * 1000
        timer['decode'] = 0.0
    f = padder.unpad(o[0])[None].float()
    return F.interpolate(f, size=(out_h, out_w), mode='bilinear', align_corners=False)[0] / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/hpc/v3_best.pth')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.75, 0.5])
    ap.add_argument('--limit', type=int, default=120)
    args = ap.parse_args()

    dev = torch.device('cuda'); amp = True
    pairs = build_vkitti2_val_pairs(args.dataset_root, ['Scene18', 'Scene20'])
    step = max(1, len(pairs) // args.limit)
    pairs = pairs[::step][:args.limit]
    print(f'{len(pairs)} pairs')

    v3 = make(args.checkpoint, True, dev)
    v2 = make(args.v2_checkpoint, False, dev)

    acc = {}
    for name in ['v3', 'v2']:
        for sc in args.scales:
            acc[(name, sc)] = dict(s=0.0, n=0, a1=0, a3=0, ms=[], co=[], de=[])

    for p1, p2, pf in tqdm(pairs):
        i1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        i2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(i1).permute(2, 0, 1).float()[None].to(dev)
        t2 = torch.from_numpy(i2).permute(2, 0, 1).float()[None].to(dev)
        gt, valid = read_vkitti2_flow(pf)
        gt = gt.to(dev); v = valid.to(dev).bool()
        H, W = gt.shape[-2:]

        for name, fn, model in (('v3', v3_at_scale, v3), ('v2', v2_at_scale, v2)):
            for sc in args.scales:
                tm = {}
                pred = fn(model, t1, t2, sc, H, W, amp, tm)
                e = torch.norm(pred - gt, dim=0)[v]
                d = acc[(name, sc)]
                d['s'] += e.sum().item(); d['n'] += e.numel()
                d['a1'] += (e < 1).sum().item(); d['a3'] += (e < 3).sum().item()
                d['ms'].append(tm['coarse'] + tm['decode'])
                d['co'].append(tm['coarse']); d['de'].append(tm['decode'])

    print(f'\nCoarse pass at reduced resolution, output at full resolution')
    print(f'VKITTI2 Scene18+20, {len(pairs)} pairs, scored against full-res ground truth\n')
    print(f'{"model":6s} {"scale":>6s} {"EPE":>8s} {"1px%":>7s} {"3px%":>7s} '
          f'{"coarse":>8s} {"decode":>8s} {"total":>8s} {"FPS":>6s}')
    print('-' * 72)
    for name in ['v2', 'v3']:
        for sc in args.scales:
            d = acc[(name, sc)]
            ms = np.mean(d['ms'][5:]); co = np.mean(d['co'][5:]); de = np.mean(d['de'][5:])
            print(f'{name:6s} {sc:6.2f} {d["s"]/d["n"]:8.3f} {d["a1"]/d["n"]*100:7.2f} '
                  f'{d["a3"]/d["n"]*100:7.2f} {co:8.1f} {de:8.1f} {ms:8.1f} {1000/ms:6.1f}')
        print()


if __name__ == '__main__':
    main()
