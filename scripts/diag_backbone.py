"""Is the v3 checkpoint's frozen front end still identical to v2's?

Two candidate explanations for v3's coarse flow diverging from v2's on the same
input:
  A) the v3 checkpoint's backbone/refinement weights drifted during training
  B) infer_coarse_state() computes something different from forward()

Test A: diff the shared weights directly.
Test B: load the v3 checkpoint into a use_implicit=False model and run v2's
        forward path. If that reproduces v2's good result, the weights are fine
        and infer_coarse_state is at fault; if it reproduces the bad result,
        the weights drifted.

    python3 scripts/diag_backbone.py
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from NeuFlow.neuflow import NeuFlow
from utils.load_model import my_load_weights, load_with_new_keys
from eval_spring_4k import (build_pairs, load_imgs, read_flo5_full, score,
                            v2_dense_upscaled)
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

    # ---- TEST A: are the shared weights identical? -------------------------
    sd_v3 = my_load_weights(args.checkpoint)
    sd_v2 = my_load_weights(args.v2_checkpoint)
    shared = [k for k in sd_v2 if k in sd_v3
              and not k.startswith(('conv_s8', 'upsample_s8', 'implicit_decoder_module'))]
    diffs = []
    for k in shared:
        a, b = sd_v2[k].float(), sd_v3[k].float()
        if a.shape != b.shape:
            diffs.append((k, float('inf'))); continue
        d = (a - b).abs().max().item()
        if d > 0:
            diffs.append((k, d))
    print(f'TEST A: {len(shared)} shared tensors, {len(diffs)} differ from v2')
    for k, d in sorted(diffs, key=lambda x: -x[1])[:8]:
        print(f'   {k:60s} max|diff| {d:.6g}')
    if not diffs:
        print('   -> frozen front end is bit-identical to v2')

    # ---- TEST B: run v2's forward path with the v3 checkpoint --------------
    print('\nTEST B: v3 checkpoint loaded into the v2 (convex upsampler) path.')
    print('        If EPE matches v2, the weights are fine and infer_coarse_state differs.\n')
    m_hybrid = NeuFlow(use_implicit=False).to(dev)
    load_with_new_keys(m_hybrid, sd_v3,
                       missing_ok_substrings=['conv_s8', 'upsample_s8'],
                       unexpected_ok_substrings=['implicit_decoder_module', 'win_proj_'])
    m_hybrid.eval()
    m_v2 = NeuFlow(use_implicit=False).to(dev)
    load_with_new_keys(m_v2, sd_v2, missing_ok_substrings=[],
                       unexpected_ok_substrings=[])
    m_v2.eval()
    m_v3 = NeuFlow(use_implicit=True, head_mode='convex').to(dev)
    load_with_new_keys(m_v3, sd_v3,
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    m_v3.eval()

    for p1, p2, pf in build_pairs(args.root, args.n):
        ta, tb = load_imgs(p1, p2, dev)
        gt = read_flo5_full(pf)
        H4, W4 = gt.shape[:2]

        # coarse flow straight out of each model's forward-path internals
        padder = frame_utils.InputPadder(ta.shape, padding_factor=16)
        pa, pb = padder.pad(ta, tb)
        m_v3.init_bhwd(1, pa.shape[-2], pa.shape[-1], dev, amp=amp)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
            st = m_v3.infer_coarse_state(pa, pb)
        c_ics = st['coarse_flow_s8'].float()

        f_hy = v2_dense_upscaled(m_hybrid, ta, tb, H4, W4, amp)
        f_v2 = v2_dense_upscaled(m_v2, ta, tb, H4, W4, amp)

        r_hy = score(f_hy, gt, 1.0); r_v2 = score(f_v2, gt, 1.0)
        print(f'{os.path.basename(pf)}')
        print(f'   v2 weights,  v2 forward path : EPE {r_v2["sum"]/r_v2["n"]:8.3f}')
        print(f'   v3 weights,  v2 forward path : EPE {r_hy["sum"]/r_hy["n"]:8.3f}')
        print(f'   v3 infer_coarse_state |flow| : mean {(c_ics*8).norm(dim=1).mean():8.2f} '
              f'max {(c_ics*8).norm(dim=1).max():8.1f}')


if __name__ == '__main__':
    main()
