"""TartanAir loader, with a convention gate.

TartanAir stores flow as .npy alongside a separate mask .npy whose values encode
several distinct conditions (occlusion, leaving the field of view, moving
object). Nothing here assumes which polarity means "valid" or which channel
order the flow uses. Those are DETERMINED by measurement in check_convention(),
because the identical assumption in read_flo5 silently halved every Spring
training target until a units check caught it.

Run the gate before trusting any number:

    python3 -m data_utils.tartanair --check --root /scratch/$USER/neuflow_datasets/tartanair
"""

import os
import os.path as osp
from glob import glob

import numpy as np


def build_pairs(root, envs=None, limit=None):
    """Return [(img0, img1, flow_npy, mask_npy), ...].

    Layout after unzip: <root>/<env>/Easy/P0xx/{image_left,flow}/...
    """
    pairs = []
    for env_dir in sorted(glob(osp.join(root, '*', 'Easy'))):
        env = osp.basename(osp.dirname(env_dir))
        if envs and env not in envs:
            continue
        for traj in sorted(glob(osp.join(env_dir, 'P*'))):
            imgs = sorted(glob(osp.join(traj, 'image_left', '*.png')))
            flows = sorted(glob(osp.join(traj, 'flow', '*_flow.npy')))
            for f in flows:
                base = osp.basename(f).replace('_flow.npy', '')
                i0, i1 = base.split('_')[:2]
                p0 = osp.join(traj, 'image_left', f'{i0}_left.png')
                p1 = osp.join(traj, 'image_left', f'{i1}_left.png')
                m = f.replace('_flow.npy', '_mask.npy')
                if osp.exists(p0) and osp.exists(p1):
                    pairs.append((p0, p1, f, m if osp.exists(m) else None))
    if limit:
        step = max(1, len(pairs) // limit)
        pairs = pairs[::step][:limit]
    return pairs


def read_flow(flow_path, mask_path=None, valid_is_zero=True):
    """Flow [H, W, 2] and a boolean valid mask [H, W].

    valid_is_zero: whether mask == 0 marks usable pixels. Established by
    check_convention(), not assumed.
    """
    flow = np.load(flow_path).astype(np.float32)
    if mask_path and osp.exists(mask_path):
        m = np.load(mask_path)
        valid = (m == 0) if valid_is_zero else (m != 0)
    else:
        valid = np.ones(flow.shape[:2], dtype=bool)
    valid &= np.isfinite(flow).all(axis=-1)
    return flow, valid


def check_convention(root, v2_checkpoint='neuflow_mixed.pth', n=4):
    """Decide the mask polarity and flow channel order by measurement.

    Scores NeuFlow v2 against each candidate reading. A correct reading should
    give an EPE of order 1 px; a wrong one is typically several times the mean
    flow magnitude. If no candidate stands out clearly, the gate FAILS and this
    dataset must not be used.
    """
    import torch
    import cv2
    import sys
    sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
    from NeuFlow.neuflow import NeuFlow
    from data_utils import frame_utils
    from utils.load_model import my_load_weights, load_with_new_keys

    pairs = build_pairs(root, limit=n)
    if not pairs:
        print(f'no TartanAir pairs under {root}')
        return None
    print(f'{len(pairs)} pairs for the convention check\n')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    m = NeuFlow(use_implicit=False).to(dev)
    load_with_new_keys(m, my_load_weights(v2_checkpoint),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=[])
    m.eval()

    cands = {
        'mask==0 valid, flow (u,v)': dict(vz=True,  swap=False, neg=False),
        'mask!=0 valid, flow (u,v)': dict(vz=False, swap=False, neg=False),
        'mask==0 valid, flow (v,u)': dict(vz=True,  swap=True,  neg=False),
        'mask==0 valid, flow negated': dict(vz=True, swap=False, neg=True),
    }
    tot = {k: [0.0, 0] for k in cands}
    magsum, magn = 0.0, 0

    for p0, p1, fp, mp in pairs:
        a = cv2.cvtColor(cv2.imread(p0), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        t0 = torch.from_numpy(a).permute(2, 0, 1).float()[None].to(dev)
        t1 = torch.from_numpy(b).permute(2, 0, 1).float()[None].to(dev)
        padder = frame_utils.InputPadder(t0.shape, padding_factor=16)
        pa, pb = padder.pad(t0, t1)
        m.init_bhwd(1, pa.shape[-2], pa.shape[-1], dev, amp=amp)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
            out = m(pa, pb)[-1]
        pred = padder.unpad(out[0]).float().cpu().numpy().transpose(1, 2, 0)

        raw = np.load(fp).astype(np.float32)
        for name, c in cands.items():
            g = raw[..., ::-1].copy() if c['swap'] else raw.copy()
            if c['neg']:
                g = -g
            _, valid = read_flow(fp, mp, valid_is_zero=c['vz'])
            if valid.sum() < 100:
                continue
            e = np.linalg.norm(pred - g, axis=-1)[valid]
            tot[name][0] += float(e.sum()); tot[name][1] += int(e.size)
        _, v0 = read_flow(fp, mp, valid_is_zero=True)
        if v0.sum() > 100:
            magsum += float(np.linalg.norm(raw, axis=-1)[v0].sum()); magn += int(v0.sum())

    print(f'{"candidate reading":32s} {"v2 EPE":>10s}')
    print('-' * 44)
    res = {}
    for name, (s, cnt) in tot.items():
        if cnt == 0:
            print(f'{name:32s} {"no valid px":>10s}'); continue
        res[name] = s / cnt
        print(f'{name:32s} {res[name]:10.3f}')
    mean_mag = magsum / magn if magn else float('nan')
    print(f'\nmean GT flow magnitude: {mean_mag:.2f} px')

    if not res:
        print('\nGATE FAILED: no candidate produced valid pixels'); return None
    best = min(res, key=res.get)
    others = sorted(v for k, v in res.items() if k != best)
    clear = others and others[0] > 2.0 * res[best]
    print(f'\nbest: {best} at {res[best]:.3f} px')
    if clear and res[best] < 0.25 * mean_mag:
        print('GATE PASSED: one reading is clearly better and is small relative '
              'to the motion.')
    else:
        print('GATE FAILED: no clear winner, or the best reading is still large '
              'relative to the motion. Do not use TartanAir numbers.')
    return best


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=f'/scratch/{os.environ.get("USER","")}/neuflow_datasets/tartanair')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--n', type=int, default=4)
    a = ap.parse_args()
    if a.check:
        check_convention(a.root, a.v2_checkpoint, a.n)
    else:
        p = build_pairs(a.root, limit=10)
        print(f'{len(p)} pairs; first: {p[0] if p else None}')
