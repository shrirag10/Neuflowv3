import torch
import numpy as np
import argparse
import os.path as osp
from glob import glob
from tqdm import tqdm

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils
from load_model import my_load_weights


def read_vkitti2_flow(path):
    flow_np, valid_np = frame_utils.read_vkitti_png_flow(path)
    # flow_np: [H, W, 2] float32, valid_np: [H, W] bool
    flow = torch.from_numpy(flow_np).permute(2, 0, 1)  # [2, H, W]
    valid = torch.from_numpy(valid_np.astype(np.float32))
    return flow, valid


def build_vkitti2_val_pairs(root, val_scenes=None):
    """Use Scene18 and Scene20 as val split (held out from training clone variant)."""
    if val_scenes is None:
        val_scenes = ['Scene18', 'Scene20']

    pairs = []
    for scene in val_scenes:
        for variant in ['clone']:
            img_dir = osp.join(root, scene, variant, 'frames', 'rgb', 'Camera_0')
            flow_dir = osp.join(root, scene, variant, 'frames', 'forwardFlow', 'Camera_0')
            if not osp.isdir(img_dir) or not osp.isdir(flow_dir):
                print(f'  WARNING: missing {img_dir} or {flow_dir}')
                continue
            images = sorted(glob(osp.join(img_dir, '*.jpg')) + glob(osp.join(img_dir, '*.png')))
            flows = sorted(glob(osp.join(flow_dir, '*.png')))
            for i in range(len(flows)):
                if i + 1 < len(images):
                    pairs.append((images[i], images[i + 1], flows[i]))
    return pairs


@torch.no_grad()
def evaluate(checkpoint, dataset_root, val_scenes, padding_factor=16, implicit=True, crop=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuFlow(use_implicit=implicit).to(device)
    state_dict = my_load_weights(checkpoint)
    model.load_state_dict(state_dict, strict=not implicit)
    model.eval()

    pairs = build_vkitti2_val_pairs(dataset_root, val_scenes)
    print(f'Val pairs: {len(pairs)} from scenes {val_scenes}')

    epe_list = []
    mag_list = []

    for img1_path, img2_path, flow_path in tqdm(pairs):
        import cv2
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None]
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()[None]

        flow_gt, valid = read_vkitti2_flow(flow_path)

        if crop is not None:
            ch, cw = crop
            H_img, W_img = img1.shape[-2], img1.shape[-1]
            y0 = (H_img - ch) // 2
            x0 = (W_img - cw) // 2
            img1 = img1[:, :, y0:y0+ch, x0:x0+cw]
            img2 = img2[:, :, y0:y0+ch, x0:x0+cw]
            flow_gt = flow_gt[:, y0:y0+ch, x0:x0+cw]
            valid   = valid[y0:y0+ch, x0:x0+cw]

        padder = frame_utils.InputPadder(img1.shape, padding_factor=padding_factor)
        img1, img2 = padder.pad(img1.to(device), img2.to(device))

        H, W = img1.shape[-2], img1.shape[-1]
        model.init_bhwd(1, H, W, device)

        with torch.amp.autocast('cuda', enabled=True):
            results = model(img1, img2)

        flow_pr = padder.unpad(results[-1][0]).float().cpu()

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        val = valid.bool().view(-1)
        epe_list.append(epe.view(-1)[val].mean().item())
        mag_list.append(mag.view(-1)[val].mean().item())

    epe_arr = np.array(epe_list)
    print(f'\n--- vKITTI2 Eval ({val_scenes}) ---')
    print(f'Mean EPE : {np.mean(epe_arr):.4f} px')
    print(f'Median EPE: {np.median(epe_arr):.4f} px')
    print(f'Mean flow mag: {np.mean(mag_list):.4f} px')
    print(f'1px acc  : {(epe_arr < 1.0).mean() * 100:.2f}%')
    print(f'3px acc  : {(epe_arr < 3.0).mean() * 100:.2f}%')
    return np.mean(epe_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_root', default='datasets/vkitti2')
    parser.add_argument('--val_scenes', nargs='+', default=['Scene18', 'Scene20'])
    parser.add_argument('--padding_factor', type=int, default=16)
    parser.add_argument('--no_implicit', action='store_true', help='Use convex upsampler (baseline)')
    parser.add_argument('--crop', type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help='Center-crop images before eval, e.g. --crop 256 512')
    args = parser.parse_args()

    evaluate(args.checkpoint, args.dataset_root, args.val_scenes, args.padding_factor,
             implicit=not args.no_implicit, crop=args.crop)
