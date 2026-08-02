"""Spring 4K evaluation: query flow ABOVE the input resolution.

Spring provides ground truth at 3840x2160 for 1920x1080 input. That makes it
the one benchmark where the two architectures are not doing the same thing:

  v3  is asked for flow AT the 4K grid positions. The decoder evaluates there
      natively, because a query is just a coordinate.
  v2  can only produce 1920x1080 and must be bilinearly upscaled to 4K. That
      upscaling is not the network answering; it is interpolation afterwards.

So this is a capability comparison, not a margin. The honest framing is: both
are scored against the same 4K ground truth, and v2's row is the best that a
fixed-resolution network can offer.

Ground-truth units: the stored .flo5 sits on a 2x grid but its VALUES are
displacements in input-resolution pixels, so predictions are compared to it
directly with no rescaling. This was established by --check_units (v2 scores
0.644 px against the raw GT and 3.320 px against a halved one) and contradicted
the comment previously in read_flo5, which had been halving Spring's targets.

    python3 scripts/eval_spring_4k.py --check_units --limit 3
    python3 scripts/eval_spring_4k.py --checkpoint <ckpt> --limit 50
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
import os.path as osp
from tqdm import tqdm

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils
from utils.load_model import my_load_weights, load_with_new_keys


def read_flo5_full(path):
    """Ground truth at its native 2x resolution, unmodified. NaN = invalid."""
    import h5py
    with h5py.File(path, 'r') as f:
        return f['flow'][()].astype(np.float32)


def build_pairs(root, limit=None):
    pairs = []
    for seq in sorted(os.listdir(root)):
        d = osp.join(root, seq)
        if not osp.isdir(d):
            continue
        imgs = sorted(glob(osp.join(d, 'frame_left', '*.png')))
        flos = sorted(glob(osp.join(d, 'flow_FW_left', '*.flo5')))
        for i in range(len(flos)):
            if i + 1 < len(imgs):
                pairs.append((imgs[i], imgs[i + 1], flos[i]))
    if limit:
        # spread the sample across sequences rather than taking one scene
        step = max(1, len(pairs) // limit)
        pairs = pairs[::step][:limit]
    return pairs


def load_imgs(p1, p2, dev):
    import cv2
    a = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
    ta = torch.from_numpy(a).permute(2, 0, 1).float()[None].to(dev)
    tb = torch.from_numpy(b).permute(2, 0, 1).float()[None].to(dev)
    return ta, tb


def make_model(ckpt, implicit, head='convex', dev='cuda', uncertainty=False):
    m = NeuFlow(use_implicit=implicit, head_mode=head,
                predict_uncertainty=uncertainty).to(dev)
    load_with_new_keys(m, my_load_weights(ckpt),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'] if implicit else [])
    m.eval()
    return m


@torch.no_grad()
def v3_query_at(model, ta, tb, out_h, out_w, amp, chunk_rows=64):
    """Decode on an out_h x out_w grid from a full-resolution input pair.

    Returns flow in INPUT pixel units on that grid. Query coordinates are the
    4K sample positions expressed in input-image coordinates, so this is the
    decoder genuinely answering between its input pixels, not interpolation.
    """
    padder = frame_utils.InputPadder(ta.shape, padding_factor=16)
    pa, pb = padder.pad(ta, tb)
    H, W = pa.shape[-2], pa.shape[-1]
    model.init_bhwd(1, H, W, ta.device, amp=amp)
    with torch.amp.autocast('cuda', enabled=amp):
        st = model.infer_coarse_state(pa, pb)

    in_h, in_w = ta.shape[-2], ta.shape[-1]
    # InputPadder pads left/top by these amounts; queries must be in padded coords
    pl, pt = padder._pad[0], padder._pad[2]

    xs = (torch.arange(out_w, device=ta.device, dtype=torch.float32) + 0.5) \
         * (in_w / out_w) - 0.5 + pl
    out = torch.empty(1, out_h, out_w, 2, device=ta.device)
    for y0 in range(0, out_h, chunk_rows):
        y1 = min(out_h, y0 + chunk_rows)
        ys = (torch.arange(y0, y1, device=ta.device, dtype=torch.float32) + 0.5) \
             * (in_h / out_h) - 0.5 + pt
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        q = torch.stack([gx, gy], -1).reshape(1, -1, 2)
        with torch.amp.autocast('cuda', enabled=amp):
            f = model.decode_queries(st, query_coords=q)
        out[:, y0:y1] = f.reshape(1, y1 - y0, out_w, 2).float()
    return out[0].permute(2, 0, 1)          # [2, out_h, out_w], input-pixel units


@torch.no_grad()
def v2_dense_upscaled(model, ta, tb, out_h, out_w, amp):
    """v2's only option: predict at input resolution, then interpolate up."""
    padder = frame_utils.InputPadder(ta.shape, padding_factor=16)
    pa, pb = padder.pad(ta, tb)
    model.init_bhwd(1, pa.shape[-2], pa.shape[-1], ta.device, amp=amp)
    with torch.amp.autocast('cuda', enabled=amp):
        out = model(pa, pb)[-1]
    f = padder.unpad(out[0])[None].float()
    return F.interpolate(f, size=(out_h, out_w), mode='bilinear',
                         align_corners=False)[0]


def score(pred_input_units, gt_4k, scale):
    """Both are in input-pixel units, so scale is 1.0 for real evaluation.
    The argument exists only so --check_units can test the alternative."""
    pred = pred_input_units * scale
    gt = torch.from_numpy(gt_4k).permute(2, 0, 1).to(pred.device)
    valid = torch.isfinite(gt).all(dim=0)
    if valid.sum() == 0:
        return None
    e = torch.norm(pred - torch.nan_to_num(gt), dim=0)[valid]
    return dict(sum=e.sum().item(), n=e.numel(),
                a1=(e < 1).sum().item(), a3=(e < 3).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=f'/scratch/{os.environ.get("USER","")}/neuflow_datasets/spring/train')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--head', default='convex')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--uncertainty', action='store_true',
                    help='checkpoint was trained with the uncertainty head (11 outputs)')
    ap.add_argument('--check_units', action='store_true',
                    help='verify the GT scale convention, then exit')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    pairs = build_pairs(args.root, args.limit)
    if not pairs:
        print(f'no Spring pairs under {args.root}'); return 1
    print(f'{len(pairs)} Spring pairs from {args.root}')

    # ---- units check -------------------------------------------------------
    if args.check_units:
        m = make_model(args.v2_checkpoint, implicit=False, dev=dev)
        print('\nIs the stored 4K ground truth in 4K pixel units or input pixel units?')
        print('Comparing v2 (which predicts in input units) against both readings.\n')
        s1 = s2 = 0.0
        for p1, p2, pf in pairs[:3]:
            ta, tb = load_imgs(p1, p2, dev)
            gt = read_flo5_full(pf)
            H4, W4 = gt.shape[:2]
            pred = v2_dense_upscaled(m, ta, tb, H4, W4, amp)
            a = score(pred, gt, scale=2.0)     # GT in 4K units
            b = score(pred, gt, scale=1.0)     # GT already in input units
            s1 += a['sum'] / a['n']; s2 += b['sum'] / b['n']
            print(f'  {osp.basename(pf)}  input {ta.shape[-2]}x{ta.shape[-1]}  '
                  f'GT {H4}x{W4}   EPE(x2)={a["sum"]/a["n"]:7.3f}   '
                  f'EPE(x1)={b["sum"]/b["n"]:7.3f}')
        print(f'\n  mean EPE assuming GT in 4K units  (pred x2): {s1/3:.3f}')
        print(f'  mean EPE assuming GT in input units (pred x1): {s2/3:.3f}')
        print(f'\n  -> the smaller value is the correct convention: '
              f'{"4K units (x2)" if s1 < s2 else "INPUT units (x1), which is what read_flo5 now does"}')
        return 0

    if not args.checkpoint:
        print('--checkpoint required (or use --check_units)'); return 1

    # ---- the comparison ----------------------------------------------------
    v3 = make_model(args.checkpoint, implicit=True, head=args.head, dev=dev,
                    uncertainty=args.uncertainty)
    v2 = make_model(args.v2_checkpoint, implicit=False, dev=dev)

    acc = {k: dict(sum=0.0, n=0, a1=0, a3=0, ms=[]) for k in
           ('v3_native_4k', 'v2_upscaled_4k', 'v3_1080p_upscaled')}

    for p1, p2, pf in tqdm(pairs):
        ta, tb = load_imgs(p1, p2, dev)
        gt = read_flo5_full(pf)
        H4, W4 = gt.shape[:2]

        # v3 queried natively on the 4K grid
        if dev.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        f3 = v3_query_at(v3, ta, tb, H4, W4, amp)
        if dev.type == 'cuda': torch.cuda.synchronize()
        t_v3 = (time.perf_counter() - t0) * 1000

        # v3 at input resolution then upscaled: isolates whether native querying
        # beats interpolating v3's own output
        f3l = v3_query_at(v3, ta, tb, ta.shape[-2], ta.shape[-1], amp)
        f3l = F.interpolate(f3l[None], size=(H4, W4), mode='bilinear',
                            align_corners=False)[0]

        # v2's only option
        if dev.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        f2 = v2_dense_upscaled(v2, ta, tb, H4, W4, amp)
        if dev.type == 'cuda': torch.cuda.synchronize()
        t_v2 = (time.perf_counter() - t0) * 1000

        for key, pred, ms in (('v3_native_4k', f3, t_v3),
                              ('v2_upscaled_4k', f2, t_v2),
                              ('v3_1080p_upscaled', f3l, float('nan'))):
            r = score(pred, gt, scale=1.0)
            if r is None:
                continue
            a = acc[key]
            a['sum'] += r['sum']; a['n'] += r['n']
            a['a1'] += r['a1']; a['a3'] += r['a3']; a['ms'].append(ms)

    print(f'\nSpring, ground truth at 3840x2160 from 1920x1080 input, {len(pairs)} pairs')
    print(f'{"method":26s} {"EPE":>8s} {"1px%":>7s} {"3px%":>7s} {"ms":>8s}')
    print('-' * 60)
    rows = [('v3 queried natively at 4K', 'v3_native_4k'),
            ('v3 at 1080p then upscaled', 'v3_1080p_upscaled'),
            ('v2 at 1080p then upscaled', 'v2_upscaled_4k')]
    for label, k in rows:
        a = acc[k]
        if a['n'] == 0:
            continue
        ms = np.nanmean(a['ms']) if len(a['ms']) else float('nan')
        print(f'{label:26s} {a["sum"]/a["n"]:8.3f} {a["a1"]/a["n"]*100:7.2f} '
              f'{a["a3"]/a["n"]*100:7.2f} {ms:8.1f}')
    print('\nv2 has no other option: its output size is fixed at the input size.')
    print('The middle row is the control -- it isolates native querying from')
    print('interpolation, using the same v3 weights.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
