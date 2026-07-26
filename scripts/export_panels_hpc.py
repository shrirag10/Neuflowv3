"""Cluster-runnable panel exporter for the new HPC checkpoints (big18, uncG,
grandmix). Same raw-panel style as export_panels.py (no titles baked in --
composed natively in the deck), but CLI-driven so it works against
/scratch/$USER paths and any checkpoint/head/uncertainty combination.

Usage (per checkpoint):
    python3 scripts/export_panels_hpc.py \
        --checkpoint /scratch/$USER/neuflow_ckpts/big18_v3dev/step_100000.pth \
        --tag big18 --dataset_root /scratch/$USER/neuflow_datasets/vkitti2

    python3 scripts/export_panels_hpc.py \
        --checkpoint /scratch/$USER/neuflow_ckpts/uncertainty_G/step_100000.pth \
        --tag uncG --uncertainty --dataset_root /scratch/$USER/neuflow_datasets/vkitti2 --skip_v2

Writes results/panels_hpc/{scene_input,scene_gt,scene_v2,scene_v3_<tag>}.png
and prints EPE for both v2 and the given checkpoint on this scene.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import cv2
import numpy as np
import torch

from NeuFlow.neuflow import NeuFlow
from utils.load_model import my_load_weights, load_with_new_keys
from data_utils import frame_utils, flow_viz

DEVICE = torch.device('cuda')
OUT = 'results/panels_hpc'


def save(name, rgb):
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(f'{OUT}/{name}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def dense_flow(model, img1, img2):
    padder = frame_utils.InputPadder(img1.shape, padding_factor=16)
    a, b = padder.pad(img1.to(DEVICE), img2.to(DEVICE))
    model.init_bhwd(1, a.shape[-2], a.shape[-1], DEVICE)
    H, W = a.shape[-2], a.shape[-1]
    with torch.no_grad(), torch.amp.autocast('cuda'):
        if not model.use_implicit:
            out = model(a, b)[-1]
        else:
            state = model.infer_coarse_state(a, b)
            ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE, dtype=torch.float32),
                                    torch.arange(W, device=DEVICE, dtype=torch.float32), indexing='ij')
            coords = torch.stack([xs, ys], -1).reshape(1, -1, 2)
            chunks = [model.decode_queries(state, query_coords=coords[:, i:i + 65536])
                      for i in range(0, coords.shape[1], 65536)]
            out = torch.cat(chunks, dim=1).reshape(1, H, W, 2).permute(0, 3, 1, 2)
    return padder.unpad(out[0]).float().cpu().permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dataset_root', default='datasets/vkitti2')
    ap.add_argument('--v2_checkpoint', default='neuflow_mixed.pth')
    ap.add_argument('--uncertainty', action='store_true')
    ap.add_argument('--skip_v2', action='store_true', help='v2/gt/input panels already exist')
    ap.add_argument('--scene', default='Scene18')
    ap.add_argument('--frame', default='00100')
    args = ap.parse_args()

    d = args.dataset_root
    p1 = f'{d}/{args.scene}/clone/frames/rgb/Camera_0/rgb_{args.frame}.jpg'
    p2 = f'{d}/{args.scene}/clone/frames/rgb/Camera_0/rgb_{int(args.frame)+1:05d}.jpg'
    pf = f'{d}/{args.scene}/clone/frames/forwardFlow/Camera_0/flow_{args.frame}.png'

    img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
    gt, valid = frame_utils.read_vkitti_png_flow(pf)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None]
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float()[None]

    def epe(f):
        e = np.linalg.norm(f - gt, axis=-1) * valid
        return e.sum() / valid.sum()

    if not args.skip_v2:
        save('scene_input', img1)
        save('scene_gt', flow_viz.flow_to_image(gt))
        v2 = NeuFlow(use_implicit=False).to(DEVICE)
        load_with_new_keys(v2, my_load_weights(args.v2_checkpoint),
                           missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                           unexpected_ok_substrings=[])
        v2.eval()
        f2 = dense_flow(v2, t1, t2)
        save('scene_v2', flow_viz.flow_to_image(f2))
        print(f'v2 EPE {epe(f2):.3f}')
        del v2
        torch.cuda.empty_cache()

    v3 = NeuFlow(use_implicit=True, head_mode='convex', predict_uncertainty=args.uncertainty).to(DEVICE)
    load_with_new_keys(v3, my_load_weights(args.checkpoint),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    v3.eval()
    f3 = dense_flow(v3, t1, t2)
    save(f'scene_v3_{args.tag}', flow_viz.flow_to_image(f3))
    print(f'v3_{args.tag} EPE {epe(f3):.3f}')


if __name__ == '__main__':
    main()
