"""Export raw, title-free image panels for native slide composition.

Writes results/panels/<stage>_<kind>.png and prints EPE values to paste
into the deck builder.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import torch

from make_stage_visuals import load_v3, load_v2, dense_flow
from data_utils import frame_utils, flow_viz

OUT = 'results/panels'
SCENE = ('datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00100.jpg',
         'datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_00101.jpg',
         'datasets/vkitti2/Scene18/clone/frames/forwardFlow/Camera_0/flow_00100.png')
CHAIRS_ID = 6

STAGES = [
    ('untrained', None),
    ('vkitti2', 'checkpoints/neuflowv3_v2dev/step_015000.pth'),
    ('mixed', 'checkpoints/neuflowv3_mix/step_030000.pth'),
]


def save(name, rgb):
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(f'{OUT}/{name}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    img1 = cv2.cvtColor(cv2.imread(SCENE[0]), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(SCENE[1]), cv2.COLOR_BGR2RGB)
    gt, valid = frame_utils.read_vkitti_png_flow(SCENE[2])
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None]
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float()[None]

    save('scene_input', img1)
    save('scene_gt', flow_viz.flow_to_image(gt))

    v2 = load_v2()
    f2 = dense_flow(v2, t1, t2)
    e2 = (np.linalg.norm(f2 - gt, axis=-1) * valid).sum() / valid.sum()
    save('scene_v2', flow_viz.flow_to_image(f2))
    print(f'scene v2 EPE {e2:.2f}')
    del v2
    torch.cuda.empty_cache()

    for tag, ckpt in STAGES:
        v3 = load_v3(ckpt)
        f3 = dense_flow(v3, t1, t2)
        e3 = (np.linalg.norm(f3 - gt, axis=-1) * valid).sum() / valid.sum()
        save(f'scene_v3_{tag}', flow_viz.flow_to_image(f3))
        print(f'scene v3_{tag} EPE {e3:.2f}')
        del v3
        torch.cuda.empty_cache()

    # chairs validation pair for the chairs stage
    d = 'datasets/FlyingChairs_release/data'
    c1 = cv2.cvtColor(cv2.imread(f'{d}/{CHAIRS_ID:05d}_img1.png'), cv2.COLOR_BGR2RGB)
    c2 = cv2.cvtColor(cv2.imread(f'{d}/{CHAIRS_ID:05d}_img2.png'), cv2.COLOR_BGR2RGB)
    cgt = frame_utils.readFlow(f'{d}/{CHAIRS_ID:05d}_flow.flo')
    tc1 = torch.from_numpy(c1).permute(2, 0, 1).float()[None]
    tc2 = torch.from_numpy(c2).permute(2, 0, 1).float()[None]
    save('chairs_input', c1)
    save('chairs_gt', flow_viz.flow_to_image(cgt))
    v2 = load_v2()
    f2 = dense_flow(v2, tc1, tc2)
    save('chairs_v2', flow_viz.flow_to_image(f2))
    print(f'chairs v2 EPE {np.linalg.norm(f2 - cgt, axis=-1).mean():.2f}')
    del v2
    torch.cuda.empty_cache()
    v3 = load_v3('checkpoints/neuflowv3_chairs_v2dev/step_030000.pth')
    f3 = dense_flow(v3, tc1, tc2)
    save('chairs_v3', flow_viz.flow_to_image(f3))
    print(f'chairs v3 EPE {np.linalg.norm(f3 - cgt, axis=-1).mean():.2f}')


if __name__ == '__main__':
    main()
