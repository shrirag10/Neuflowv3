"""Crop to a region of interest at FULL resolution and measure what it costs.

This is the fast-platform operation: keep full fidelity where you are looking,
spend nothing elsewhere. Distinct from reducing resolution, which loses detail
everywhere (see eval_coarse_resolution.py -- that approach fails).

The question this answers: NeuFlow finds large motion with global attention at
1/16 scale, so a crop removes context and a match lying outside the box becomes
unfindable. How much margin does an ROI crop need, and what does it cost?

Accuracy is scored ONLY inside the ROI, against the full-frame result as
reference, because accuracy outside the box is irrelevant to this use case.

    python3 scripts/eval_roi_crop.py --limit 40
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
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
def run_region(model, t1, t2, box, margin, amp, implicit):
    """Crop to box+margin, run the pipeline, return flow for the ROI only.

    box = (x0, y0, x1, y1) in full-image pixels. Returns [2, bh, bw] covering
    exactly the ROI, plus the coarse-pass latency and the cropped area.
    """
    H, W = t1.shape[-2], t1.shape[-1]
    x0, y0, x1, y1 = box
    cx0 = max(0, x0 - margin); cy0 = max(0, y0 - margin)
    cx1 = min(W, x1 + margin); cy1 = min(H, y1 + margin)

    a = t1[:, :, cy0:cy1, cx0:cx1]
    b = t2[:, :, cy0:cy1, cx0:cx1]
    padder = frame_utils.InputPadder(a.shape, padding_factor=16)
    pa, pb = padder.pad(a, b)
    model.init_bhwd(1, pa.shape[-2], pa.shape[-1], t1.device, amp=amp)

    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.amp.autocast('cuda', enabled=amp):
        if implicit:
            st = model.infer_coarse_state(pa, pb)
            out = model.decode_dense_fast(st, stride=2)
        else:
            out = model(pa, pb)[-1]
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000

    f = padder.unpad(out[0]).float()          # [2, ch, cw] in crop coords
    # slice out the ROI itself (flow values are displacements, no offset needed)
    f = f[:, y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]
    return f, ms, (cy1 - cy0) * (cx1 - cx0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/hpc/v3_best.pth')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--roi', type=int, default=192, help='ROI side in pixels')
    ap.add_argument('--margins', type=int, nargs='+', default=[0, 16, 32, 64, 128])
    ap.add_argument('--limit', type=int, default=40)
    args = ap.parse_args()

    dev = torch.device('cuda'); amp = True
    pairs = build_vkitti2_val_pairs(args.dataset_root, ['Scene18', 'Scene20'])
    step = max(1, len(pairs) // args.limit)
    pairs = pairs[::step][:args.limit]
    print(f'{len(pairs)} pairs, ROI {args.roi}x{args.roi}')

    models = {'v3': (make(args.checkpoint, True, dev), True),
              'v2': (make(args.v2_checkpoint, False, dev), False)}

    # accumulators: (model, margin) and (model, 'full')
    acc = {}
    def slot(k):
        return acc.setdefault(k, dict(s=0.0, n=0, ms=[], area=[], big=0, bign=0))

    for p1, p2, pf in tqdm(pairs):
        i1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        i2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(i1).permute(2, 0, 1).float()[None].to(dev)
        t2 = torch.from_numpy(i2).permute(2, 0, 1).float()[None].to(dev)
        gt, valid = read_vkitti2_flow(pf)
        gt = gt.to(dev); vmask = valid.to(dev).bool()
        H, W = gt.shape[-2:]

        # ROI centred on the strongest corner cluster: "something worth flowing"
        g = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
        pts = cv2.goodFeaturesToTrack(g, maxCorners=200, qualityLevel=0.01, minDistance=10)
        cxy = pts.reshape(-1, 2).mean(0) if pts is not None else np.array([W / 2, H / 2])
        r = args.roi // 2
        x0 = int(np.clip(cxy[0] - r, 0, W - args.roi)); x1 = x0 + args.roi
        y0 = int(np.clip(cxy[1] - r, 0, H - args.roi)); y1 = y0 + args.roi
        box = (x0, y0, x1, y1)

        gt_roi = gt[:, y0:y1, x0:x1]
        v_roi = vmask[y0:y1, x0:x1]
        mag = torch.norm(gt_roi, dim=0)[v_roi]

        for name, (model, implicit) in models.items():
            # full-frame reference
            f, ms, area = run_region(model, t1, t2, box, max(H, W), amp, implicit)
            e = torch.norm(f - gt_roi, dim=0)[v_roi]
            d = slot((name, 'full'))
            d['s'] += e.sum().item(); d['n'] += e.numel()
            d['ms'].append(ms); d['area'].append(area)

            for mg in args.margins:
                f, ms, area = run_region(model, t1, t2, box, mg, amp, implicit)
                e = torch.norm(f - gt_roi, dim=0)[v_roi]
                d = slot((name, mg))
                d['s'] += e.sum().item(); d['n'] += e.numel()
                d['ms'].append(ms); d['area'].append(area)
                # match failures: large error where motion is large
                big = mag > 8
                if big.any():
                    d['big'] += (e[big] > 3).sum().item(); d['bign'] += big.sum().item()

    full_area = H * W
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    print(f'\nROI {args.roi}x{args.roi} at full resolution, error scored inside the ROI only')
    print(f'VKITTI2 Scene18+20, {len(pairs)} pairs, {H}x{W} frames')
    print(f'device: {dev_name}')
    print('Crop savings depend on whether the device is compute bound or launch\n'
          'bound at this size: shrinking the area cannot shrink a fixed number of\n'
          'kernel launches. Report the speedup per device, not as one figure.\n')
    print(f'{"model":6s} {"margin":>7s} {"area%":>7s} {"EPE":>8s} {"vs full":>9s} '
          f'{"ms":>7s} {"speedup":>8s} {"fail%":>7s}')
    print('-' * 66)
    for name in ['v2', 'v3']:
        ref = acc[(name, 'full')]
        ref_epe = ref['s'] / ref['n']; ref_ms = np.mean(ref['ms'][3:])
        print(f'{name:6s} {"full":>7s} {100.0:7.1f} {ref_epe:8.3f} {"-":>9s} '
              f'{ref_ms:7.1f} {"1.00x":>8s} {"-":>7s}')
        for mg in args.margins:
            d = acc[(name, mg)]
            epe = d['s'] / d['n']; ms = np.mean(d['ms'][3:])
            ar = np.mean(d['area']) / full_area * 100
            fail = d['big'] / d['bign'] * 100 if d['bign'] else float('nan')
            print(f'{name:6s} {mg:7d} {ar:7.1f} {epe:8.3f} {epe-ref_epe:+9.3f} '
                  f'{ms:7.1f} {ref_ms/ms:7.2f}x {fail:7.1f}')
        print()
    print('fail% = fraction of large-motion pixels (GT > 8 px) with error > 3 px')


if __name__ == '__main__':
    main()
