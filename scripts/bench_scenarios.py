"""Three fast-platform scenarios: what does it cost to flow only part of the image?

The scenarios, as posed:

  S1  first encounter   one ROI enters the field of view
  S2  turn              the tracked ROI now overlaps a second object
  S3  new object        a second, disjoint ROI appears in a frame already started

Four policies, measured on each:

  v2_full   NeuFlow v2 on the whole frame (its only native mode)
  v2_crop   NeuFlow v2 on ROI + margin
  v3_full   v3 full-frame coarse pass, decode only inside the ROIs
  v3_crop   v3 coarse pass on ROI + margin, decode inside

  v3_sparse N corner points inside the ROI rather than a dense field. This is
            the mode the queryable decoder is for, and the only one where the
            marginal cost of a second ROI is the flat ~2.5 ms figure: decode is
            launch-overhead bound only up to about 2,048 points. A dense
            192x192 ROI is 36,864 points and is compute bound instead.

All v3 policies decode at the same stride so the comparison is like-for-like;
an earlier version had v3_full at stride 1 and v3_crop at stride 2, which made
v3_full look 4x more expensive than it is.

The number that separates them is not total cost but the MARGINAL cost of the
second ROI. v2_full has already computed everything, so its marginal cost is
zero -- but it paid the most up front. The cropped policies must run a whole new
coarse pass for an ROI outside their crop. v3_full pays only a decode.

Accuracy is scored inside the ROIs only; error elsewhere is irrelevant here.

    python3 scripts/bench_scenarios.py --limit 40
    python3 scripts/bench_scenarios.py --sanity      # harness check, see below
"""

import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2
from tqdm import tqdm

from data_utils import frame_utils
from eval_vkitti2 import build_vkitti2_val_pairs, read_vkitti2_flow
from flow_engine import FlowEngine

POLICIES = ['v2_full', 'v2_crop', 'v3_full', 'v3_crop', 'v3_sparse']


# --------------------------------------------------------------------------- v2 crop
def v2_crop_region(eng, img0, img1, box, margin):
    """v2 on box+margin. v2 has no coarse/decode split, so one number."""
    H, W = img0.shape[:2]
    x0, y0, x1, y1 = box
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(W, x1 + margin), min(H, y1 + margin)
    a = eng._to_tensor(img0[cy0:cy1, cx0:cx1])
    b = eng._to_tensor(img1[cy0:cy1, cx0:cx1])
    padder = frame_utils.InputPadder(a.shape, padding_factor=16)
    a, b = padder.pad(a, b)
    eng._prep(eng.v2, a.shape[-2], a.shape[-1])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=eng.amp):
        out = eng.v2(a, b)[-1]
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
    dense = padder.unpad(out[0]).float().cpu().numpy().transpose(1, 2, 0)
    return dense[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0], ms, (cx1 - cx0) * (cy1 - cy0)


def v3_crop_dense(eng, img0, img1, box, margin, stride=2):
    """v3 on box+margin, returning the dense flow for the WHOLE crop.

    crop_region() slices to the requested box and does not expose the coarse
    state, so a second box inside the same crop cannot be decoded from it. The
    crop is densely decoded anyway, so returning all of it is both cheaper and
    honest: anything inside the crop is already answered.
    """
    H, W = img0.shape[:2]
    x0, y0, x1, y1 = box
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(W, x1 + margin), min(H, y1 + margin)
    a = eng._to_tensor(img0[cy0:cy1, cx0:cx1])
    b = eng._to_tensor(img1[cy0:cy1, cx0:cx1])
    padder = frame_utils.InputPadder(a.shape, padding_factor=16)
    a, b = padder.pad(a, b)
    eng._prep(eng.v3, a.shape[-2], a.shape[-1])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=eng.amp):
        st = eng.v3.infer_coarse_state(a, b)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=eng.amp):
        dense = eng.v3.decode_dense_fast(st, stride=stride)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    dense = padder.unpad(dense[0]).float().cpu().numpy().transpose(1, 2, 0)
    return dense, (cx0, cy0, cx1, cy1), (t1 - t0) * 1000, (t2 - t1) * 1000


# --------------------------------------------------------------------------- helpers
def epe_in(flow, gt, valid, box):
    x0, y0, x1, y1 = box
    g = gt[:, y0:y1, x0:x1].permute(1, 2, 0).cpu().numpy()
    v = valid[y0:y1, x0:x1].cpu().numpy().astype(bool)
    if v.sum() == 0:
        return None, None, 0
    e = np.linalg.norm(flow - g, axis=-1)[v]
    mag = np.linalg.norm(g, axis=-1)[v]
    big = mag > 8
    fail = float((e[big] > 3).mean()) if big.any() else np.nan
    return float(e.mean()), fail, int(v.sum())


def boxes_for(scenario, img, roi):
    """ROI geometry for each scenario, on a real frame.

    Box 1 sits on the strongest corner cluster: 'something worth flowing'.
    S2 places box 2 to overlap box 1 by ~40% of its area; S3 places it far away.
    """
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pts = cv2.goodFeaturesToTrack(g, maxCorners=200, qualityLevel=0.01, minDistance=10)
    c = pts.reshape(-1, 2).mean(0) if pts is not None else np.array([W / 2, H / 2])
    x0 = int(np.clip(c[0] - roi // 2, 0, W - roi)); y0 = int(np.clip(c[1] - roi // 2, 0, H - roi))
    b1 = (x0, y0, x0 + roi, y0 + roi)
    if scenario == 'S1':
        return [b1]
    if scenario == 'S2':                       # 60% shift => ~40% overlap
        sx = int(np.clip(x0 + int(roi * 0.6), 0, W - roi))
        return [b1, (sx, y0, sx + roi, y0 + roi)]
    # S3: disjoint, pushed to the far side of the frame
    sx = W - roi if x0 < W // 2 else 0
    return [b1, (sx, y0, sx + roi, y0 + roi)]


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def inside(inner, outer, margin):
    return (inner[0] >= outer[0] - margin and inner[1] >= outer[1] - margin and
            inner[2] <= outer[2] + margin and inner[3] <= outer[3] + margin)


# --------------------------------------------------------------------------- run one
def run_policy(pol, eng, img0, img1, boxes, margin, gt, valid, key,
               stride=2, n_sparse=800):
    """Returns (first_ms, marginal_ms, [epe per box], [fail per box], area_px).

    first_ms    cost of answering box 1
    marginal_ms ADDITIONAL cost of answering box 2 given box 1 was answered
    """
    flows, first, marginal, area = [], 0.0, 0.0, 0
    sparse_epes, sparse_n = [], []

    if pol == 'v2_full':
        dense, ms = eng.full_frame_v2(img0, img1)
        first = ms
        marginal = 0.0                          # already has the whole frame
        area = img0.shape[0] * img0.shape[1]
        for b in boxes:
            flows.append(dense[b[1]:b[3], b[0]:b[2]])

    elif pol == 'v2_crop':
        for i, b in enumerate(boxes):
            f, ms, ar = v2_crop_region(eng, img0, img1, b, margin)
            flows.append(f); area += ar
            if i == 0: first = ms
            else:      marginal += ms           # a whole new pass per ROI

    elif pol == 'v3_full':
        eng.coarse(img0, img1, key=key)
        c_ms = eng.coarse_ms
        area = img0.shape[0] * img0.shape[1]
        for i, b in enumerate(boxes):
            f, _, d_ms = eng.query_region(eng.state, b[0], b[1], b[2] - b[0], b[3] - b[1],
                                          stride=stride)
            flows.append(f)
            if i == 0: first = c_ms + d_ms
            else:      marginal += d_ms         # decode only, state is cached

    elif pol == 'v3_crop':
        # crop around box 1. A second box already covered by that crop is free,
        # because the crop was densely decoded. One outside costs a whole new pass.
        b = boxes[0]
        dense, ext, c_ms, d_ms = v3_crop_dense(eng, img0, img1, b, margin, stride=stride)
        first = c_ms + d_ms
        area += (ext[2] - ext[0]) * (ext[3] - ext[1])
        flows.append(dense[b[1] - ext[1]:b[3] - ext[1], b[0] - ext[0]:b[2] - ext[0]])
        for b2 in boxes[1:]:
            if b2[0] >= ext[0] and b2[1] >= ext[1] and b2[2] <= ext[2] and b2[3] <= ext[3]:
                flows.append(dense[b2[1] - ext[1]:b2[3] - ext[1],
                                   b2[0] - ext[0]:b2[2] - ext[0]])
                marginal += 0.0                 # already inside the decoded crop
            else:
                d2, ext2, c2, dd2 = v3_crop_dense(eng, img0, img1, b2, margin, stride=stride)
                flows.append(d2[b2[1] - ext2[1]:b2[3] - ext2[1],
                                b2[0] - ext2[0]:b2[2] - ext2[0]])
                marginal += c2 + dd2
                area += (ext2[2] - ext2[0]) * (ext2[3] - ext2[1])

    elif pol == 'v3_sparse':
        # N corner points inside each ROI: the intended use, and the regime
        # where decode cost is flat in N
        eng.coarse(img0, img1, key=key)
        c_ms = eng.coarse_ms
        area = img0.shape[0] * img0.shape[1]
        pad = eng.state.get('_padder')
        ox = pad._pad[0] if pad is not None else 0
        oy = pad._pad[2] if pad is not None else 0
        g = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        for i, b in enumerate(boxes):
            sub = g[b[1]:b[3], b[0]:b[2]]
            pts = cv2.goodFeaturesToTrack(sub, maxCorners=n_sparse,
                                          qualityLevel=0.005, minDistance=3)
            if pts is None or len(pts) < 8:
                sparse_epes.append(None); continue
            pxy = pts.reshape(-1, 2) + np.array([b[0], b[1]], dtype=np.float32)
            q = torch.from_numpy(pxy).float().to(eng.dev)[None]
            q = q + torch.tensor([ox, oy], device=eng.dev, dtype=q.dtype)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=eng.amp):
                fl = eng.v3.decode_queries(eng.state, query_coords=q)
            torch.cuda.synchronize(); d_ms = (time.perf_counter() - t0) * 1000
            if i == 0: first = c_ms + d_ms
            else:      marginal += d_ms
            # error at those points against GT
            gxy = gt[:, np.clip(pxy[:, 1].astype(int), 0, gt.shape[1] - 1),
                        np.clip(pxy[:, 0].astype(int), 0, gt.shape[2] - 1)].T
            e = torch.norm(fl[0].float() - gxy, dim=-1)
            sparse_epes.append(float(e.mean()))
            sparse_n.append(len(pxy))

    epes, fails = [], []
    if pol == 'v3_sparse':
        epes = [x for x in sparse_epes if x is not None]
        return first, marginal, epes, [], area
    for f, b in zip(flows, boxes):
        e, fa, n = epe_in(f, gt, valid, b)
        if e is not None:
            epes.append(e); fails.append(fa)
    return first, marginal, epes, fails, area


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/hpc/v3_best.pth')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--roi', type=int, default=192)
    ap.add_argument('--margin', type=int, default=32,
                    help='pixels; 32 matches the measured knee at 26.6 px mean motion')
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--stride', type=int, default=2,
                    help='decode stride for the dense v3 policies; shared so they compare')
    ap.add_argument('--n_sparse', type=int, default=800,
                    help='corner points per ROI for v3_sparse')
    ap.add_argument('--sanity', action='store_true',
                    help='full-frame-sized margin: every policy must agree with '
                         'the known full-frame numbers')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    pairs = build_vkitti2_val_pairs(args.dataset_root, ['Scene18', 'Scene20'])
    step = max(1, len(pairs) // args.limit)
    pairs = pairs[::step][:args.limit]

    eng = FlowEngine(args.checkpoint, args.v2_checkpoint)
    scenarios = ['S1'] if args.sanity else ['S1', 'S2', 'S3']
    margin = 10000 if args.sanity else args.margin
    print(f'{len(pairs)} pairs, ROI {args.roi}, margin {margin}'
          + ('  [SANITY: margin covers the frame, all policies should agree]' if args.sanity else ''))

    acc = {(sc, p): dict(first=[], marg=[], epe=[], fail=[], area=[])
           for sc in scenarios for p in POLICIES}
    warmed = set()

    for idx, (p1, p2, pf) in enumerate(tqdm(pairs)):
        i0 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        i1 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        gt, valid = read_vkitti2_flow(pf)
        gt = gt.to(eng.dev); valid = valid.to(eng.dev)
        H, W = i0.shape[:2]
        if (H, W) not in warmed:
            eng.warmup(H, W); warmed.add((H, W))

        for sc in scenarios:
            boxes = boxes_for(sc, i0, args.roi)
            # warm every distinct crop shape once: a cold shape pays cuDNN
            # autotune and would be reported as pipeline cost
            for b in boxes:
                ch = min(H, b[3] + margin) - max(0, b[1] - margin)
                cw = min(W, b[2] + margin) - max(0, b[0] - margin)
                if (ch, cw) not in warmed:
                    eng.warmup(ch, cw); warmed.add((ch, cw))

            for pol in POLICIES:
                f, m, epes, fails, area = run_policy(
                    pol, eng, i0, i1, boxes, margin, gt, valid, key=f'{idx}',
                    stride=args.stride, n_sparse=args.n_sparse)
                d = acc[(sc, pol)]
                d['first'].append(f); d['marg'].append(m)
                d['epe'] += epes; d['fail'] += [x for x in fails if not np.isnan(x)]
                d['area'].append(area)

    full_px = H * W
    print()
    for sc in scenarios:
        n_roi = len(boxes_for(sc, i0, args.roi))
        print(f'--- {sc} ({n_roi} ROI{"s" if n_roi > 1 else ""}) '
              f'{"first encounter" if sc=="S1" else "turn, overlapping" if sc=="S2" else "new object, disjoint"}')
        print(f'{"policy":9s} {"first ms":>9s} {"+2nd ROI":>9s} {"total":>8s} '
              f'{"area%":>7s} {"EPE":>7s} {"fail%":>7s}')
        for pol in POLICIES:
            d = acc[(sc, pol)]
            fi = np.median(d['first']); mg = np.median(d['marg'])
            ar = np.mean(d['area']) / full_px * 100
            ep = np.mean(d['epe']) if d['epe'] else float('nan')
            fl = np.mean(d['fail']) * 100 if d['fail'] else float('nan')
            print(f'{pol:9s} {fi:9.1f} {mg:9.2f} {fi+mg:8.1f} {ar:7.1f} {ep:7.3f} {fl:7.1f}')
        print()

    if args.json:
        out = {f'{sc}|{pol}': {k: (list(map(float, v)) if isinstance(v, list) else v)
                               for k, v in acc[(sc, pol)].items()}
               for sc in scenarios for pol in POLICIES}
        json.dump(out, open(args.json, 'w'), indent=1)
        print('wrote', args.json)


if __name__ == '__main__':
    main()
