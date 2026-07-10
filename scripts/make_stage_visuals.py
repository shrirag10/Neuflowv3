"""Per-stage visual comparisons: for each v3 training stage, GT vs v2 vs v3
on two different scenes. Outputs results/visuals/stage_<name>.png.

The untrained stage is constructed in place (bilinear-prior convex init),
so no checkpoint is needed for it.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils, flow_viz
from utils.load_model import my_load_weights, load_with_new_keys

DEVICE = torch.device('cuda')

SCENES = [
    ('Scene18 · highway, oncoming traffic',
     'datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00100.jpg',
     'datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00101.jpg',
     'datasets/vkitti2/Scene18/clone/frames/forwardFlow/Camera_0/flow_00100.png'),
    ('Scene20 · urban curve, parked cars',
     'datasets/vkitti2/Scene20/clone/frames/rgb/Camera_0/rgb_00050.jpg',
     'datasets/vkitti2/Scene20/clone/frames/rgb/Camera_0/rgb_00051.jpg',
     'datasets/vkitti2/Scene20/clone/frames/forwardFlow/Camera_0/flow_00050.png'),
]

STAGES = [
    ('untrained', 'No decoder training (bilinear-prior initialization)', None),
    ('vkitti2', 'Trained on VKITTI2 only (12.7k pairs)', 'checkpoints/neuflowv3_v2dev/step_015000.pth'),
    ('chairs', 'Trained on FlyingChairs only (22.2k pairs)', 'checkpoints/neuflowv3_chairs_v2dev/step_030000.pth'),
]
OUT = 'results/visuals'


def load_v3(ckpt):
    m = NeuFlow(use_implicit=True, head_mode='convex').to(DEVICE)
    src = ckpt if ckpt else 'neuflow_mixed.pth'
    load_with_new_keys(m, my_load_weights(src),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    if ckpt is None:
        head = m.implicit_decoder_module.convex_head
        torch.nn.init.zeros_(head.layers[-1].weight)
        torch.nn.init.zeros_(head.layers[-1].bias)
    m.eval()
    return m


def load_v2():
    m = NeuFlow(use_implicit=False).to(DEVICE)
    load_with_new_keys(m, my_load_weights('neuflow_mixed.pth'),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=[])
    m.eval()
    return m


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
    os.makedirs(OUT, exist_ok=True)
    v2 = load_v2()

    scene_data = []
    v2_flows = []
    for name, p1, p2, pf in SCENES:
        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None]
        t2 = torch.from_numpy(cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None]
        gt, valid = frame_utils.read_vkitti_png_flow(pf)
        scene_data.append((name, img1, t1, t2, gt, valid))
        v2_flows.append(dense_flow(v2, t1, t2))
    del v2
    torch.cuda.empty_cache()

    def epe(f, gt, valid):
        e = np.linalg.norm(f - gt, axis=-1) * valid
        return e.sum() / valid.sum()

    for tag, desc, ckpt in STAGES:
        v3 = load_v3(ckpt)
        fig, axes = plt.subplots(4, 2, figsize=(16, 9.5), dpi=130)
        for col, ((name, img1, t1, t2, gt, valid), f2) in enumerate(zip(scene_data, v2_flows)):
            f3 = dense_flow(v3, t1, t2)
            rows = [
                (img1, name),
                (flow_viz.flow_to_image(gt), 'Ground truth'),
                (flow_viz.flow_to_image(f2), f'NeuFlow v2 — EPE {epe(f2, gt, valid):.2f} px'),
                (flow_viz.flow_to_image(f3), f'NeuFlow v3 ({tag}) — EPE {epe(f3, gt, valid):.2f} px'),
            ]
            for row, (im, title) in enumerate(rows):
                axes[row][col].imshow(im)
                axes[row][col].set_title(title, fontsize=11, loc='left')
                axes[row][col].axis('off')
        fig.suptitle(desc, fontsize=14, x=0.01, ha='left', fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = f'{OUT}/stage_{tag}.png'
        plt.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(path)
        del v3
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
