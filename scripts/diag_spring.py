"""Localise the Spring failure: is it the coarse pass or the decoder?

v2 and v3 share the same frozen backbone, so their coarse flow must be
identical. If the coarse flow alone scores well and the decoder scores badly,
the decoder is at fault and the 4K result says nothing about querying.

    python3 scripts/diag_spring.py
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from eval_spring_4k import (make_model, load_imgs, read_flo5_full, build_pairs,
                            v3_query_at, v2_dense_upscaled, score)
from data_utils import frame_utils


def main():
    ap = argparse.ArgumentParser()
    u = os.environ.get('USER', '')
    ap.add_argument('--root', default=f'/scratch/{u}/neuflow_datasets/spring/train')
    ap.add_argument('--checkpoint',
                    default=f'/scratch/{u}/neuflow_ckpts/v3_FlyingChairs_VKITTI2_Sintel/step_100000.pth')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--n', type=int, default=3)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    pairs = build_pairs(args.root, args.n)
    v3 = make_model(args.checkpoint, implicit=True, dev=dev)
    v2 = make_model(args.v2_checkpoint, implicit=False, dev=dev)

    for p1, p2, pf in pairs:
        ta, tb = load_imgs(p1, p2, dev)
        gt = read_flo5_full(pf)
        H4, W4 = gt.shape[:2]
        g = torch.from_numpy(gt).permute(2, 0, 1)
        m = torch.isfinite(g).all(0)
        gt_mag = g.permute(1, 2, 0)[m].norm(dim=-1)

        padder = frame_utils.InputPadder(ta.shape, padding_factor=16)
        pa, pb = padder.pad(ta, tb)
        v3.init_bhwd(1, pa.shape[-2], pa.shape[-1], dev, amp=amp)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
            st = v3.infer_coarse_state(pa, pb)

        # coarse flow alone, no decoder involved
        coarse = st['coarse_flow_s8'].float() * 8.0
        cf = padder.unpad(F.interpolate(coarse, size=pa.shape[-2:], mode='bilinear',
                                        align_corners=False)[0])
        cf4 = F.interpolate(cf[None], size=(H4, W4), mode='bilinear', align_corners=False)[0]

        d3 = v3_query_at(v3, ta, tb, ta.shape[-2], ta.shape[-1], amp)
        d3u = F.interpolate(d3[None], size=(H4, W4), mode='bilinear', align_corners=False)[0]
        f2 = v2_dense_upscaled(v2, ta, tb, H4, W4, amp)

        print(f'\n{os.path.basename(pf)}   input {tuple(ta.shape[-2:])}  GT {H4}x{W4}')
        print(f'  {"ground-truth |flow|":34s} mean {gt_mag.mean():8.2f}  '
              f'p99 {gt_mag.kthvalue(int(0.99*len(gt_mag)))[0]:8.2f}  max {gt_mag.max():8.2f}')
        for name, pred in (('coarse flow x8 (NO decoder)', cf4),
                           ('v3 decoder at 1080p', d3u),
                           ('v2 convex upsampler', f2)):
            r = score(pred, gt, 1.0)
            pm = pred.norm(dim=0)
            print(f'  {name:34s} EPE  {r["sum"]/r["n"]:8.3f}   '
                  f'|pred| mean {pm.mean():7.2f} max {pm.max():8.1f}')


if __name__ == '__main__':
    main()
