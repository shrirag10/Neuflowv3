"""Sparse-query registration demo for NeuFlow v3.

Demonstrates the queryable-flow capability: run the backbone ONCE, then ask the
implicit decoder for flow at a sparse set of textured points only (cost scales
with #queries, not image area). Those correspondences are fed to a RANSAC
homography to register frame 1 onto frame 0, and the result is shown as a
checkerboard overlay (alignment = crisp seams, misalignment = ghosting).

This proves the algorithm produces *usable* correspondences for mapping without
ever computing a dense flow field, and it is robust to the current EPE because
RANSAC rejects outliers.

Usage:
  python3 scripts/demo_registration.py \
    --checkpoint checkpoints/neuflowv3/step_010000.pth \
    --img0 <frame0> --img1 <frame1> --out results/demo_registration

Defaults to a VKITTI2 Scene18 pair if no images are given.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import cv2
import torch
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from utils.load_model import my_load_weights

IMAGE_W = 512
IMAGE_H = 256


def load_img(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.resize(bgr, (IMAGE_W, IMAGE_H))


def to_tensor(bgr, device):
    t = torch.from_numpy(bgr).permute(2, 0, 1).float()
    return t[None].to(device)


def fuse_conv_and_bn(conv, bn):
    fused = torch.nn.Conv2d(
        conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True,
    ).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


def build_model(checkpoint, device, amp):
    model = NeuFlow(use_implicit=True, head_mode='regress').to(device)  # matches this script's pre-convex-head checkpoint default
    # Pre-window checkpoints lack win_proj_* keys; __init__ already center-inits
    # them (equivalent to point-sampling), so strict=False reproduces them faithfully.
    missing, unexpected = model.load_state_dict(my_load_weights(checkpoint), strict=False)
    missing = [k for k in missing if not k.startswith('implicit_decoder_module.win_proj_')]
    if missing or unexpected:
        raise RuntimeError(f'Unexpected checkpoint mismatch.\nmissing={missing}\nunexpected={unexpected}')
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
            delattr(m, 'norm1'); delattr(m, 'norm2')
            m.forward = m.forward_fuse
    model.eval()
    model.init_bhwd(1, IMAGE_H, IMAGE_W, device, amp=amp)
    return model


def checkerboard(a, b, tile=32):
    """Interleave two images in a checkerboard pattern to reveal misalignment."""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy // tile) + (xx // tile)) % 2 == 0)[..., None]
    return np.where(mask, a, b).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/neuflowv3/step_010000.pth')
    ap.add_argument('--img0', default=None)
    ap.add_argument('--img1', default=None)
    ap.add_argument('--num_points', type=int, default=800)
    ap.add_argument('--out', default='results/demo_registration')
    args = ap.parse_args()

    if args.img0 is None or args.img1 is None:
        base = 'datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0'
        args.img0 = os.path.join(base, 'rgb_00000.jpg')
        args.img1 = os.path.join(base, 'rgb_00002.jpg')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'
    os.makedirs(args.out, exist_ok=True)

    bgr0 = load_img(args.img0)
    bgr1 = load_img(args.img1)
    t0, t1 = to_tensor(bgr0, device), to_tensor(bgr1, device)

    model = build_model(args.checkpoint, device, amp)

    # --- Pick sparse textured query points in frame 0 ---
    gray0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray0, maxCorners=args.num_points,
                                  qualityLevel=0.01, minDistance=6)
    if pts is None:
        raise RuntimeError('No corners found.')
    pts = pts.reshape(-1, 2).astype(np.float32)       # (N, 2) in (x, y)
    n_query = len(pts)

    # --- ONE backbone pass, then decode flow ONLY at those points ---
    query = torch.from_numpy(pts)[None].to(device)    # [1, N, 2]
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=amp):
        state = model.infer_coarse_state(t0, t1)
        flow_q = model.decode_queries(state, query_coords=query)   # [1, N, 2]
    flow_q = flow_q[0].float().cpu().numpy()

    p0 = pts                                           # source points (frame 0)
    p1 = pts + flow_q                                  # corresponding points (frame 1)

    # --- Robust registration: homography via RANSAC ---
    H, inliers = cv2.findHomography(p1, p0, cv2.RANSAC, ransacReprojThreshold=3.0)
    inliers = inliers.ravel().astype(bool)
    inlier_ratio = inliers.mean()

    # Reprojection error on inliers. NOTE: this is SELF-CONSISTENCY of the
    # inlier matches with the fitted homography, not flow accuracy vs ground truth.
    p1h = np.hstack([p1, np.ones((len(p1), 1), np.float32)])
    proj = (H @ p1h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    reproj_err = np.linalg.norm(proj[inliers] - p0[inliers], axis=1).mean()

    # Is the homography non-trivial (not ~identity)? Mean corner displacement.
    corners = np.array([[0, 0], [IMAGE_W, 0], [IMAGE_W, IMAGE_H], [0, IMAGE_H]],
                       np.float32).reshape(-1, 1, 2)
    warp_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    corner_shift = np.linalg.norm(warp_corners - corners.reshape(-1, 2), axis=1).mean()

    # --- Warp frame 1 into frame 0 and build overlays ---
    warped1 = cv2.warpPerspective(bgr1, H, (IMAGE_W, IMAGE_H))

    # Honest registration evidence: does warping reduce photometric error vs not warping?
    cover = warped1.sum(-1) > 0
    err_nowarp = np.abs(bgr0.astype(np.float32) - bgr1.astype(np.float32))[cover].mean()
    err_warp = np.abs(bgr0.astype(np.float32) - warped1.astype(np.float32))[cover].mean()
    photo_gain = 100.0 * (err_nowarp - err_warp) / err_nowarp
    cb_before = checkerboard(bgr0, bgr1)
    cb_after = checkerboard(bgr0, warped1)
    blend_after = cv2.addWeighted(bgr0, 0.5, warped1, 0.5, 0)

    # Correspondence visualization (inliers only, subsampled)
    corr_vis = bgr0.copy()
    idx = np.where(inliers)[0]
    for i in idx[::max(1, len(idx) // 150)]:
        a = tuple(np.round(p0[i]).astype(int))
        b = tuple(np.round(p1[i]).astype(int))
        cv2.line(corr_vis, a, b, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(corr_vis, a, 2, (0, 0, 255), -1)

    def label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (IMAGE_W, 22), (30, 30, 30), -1)
        cv2.putText(out, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    top = np.hstack([label(cb_before, 'Before: frame t & t+1 overlaid (ghosting)'),
                     label(corr_vis, f'{int(inlier_ratio*n_query)} inlier queries -> homography')])
    bot = np.hstack([label(cb_after, 'After: frame t+1 registered to t (checkerboard)'),
                     label(blend_after, 'After: 50/50 blend')])
    panel = np.vstack([top, bot])

    dense_px = IMAGE_W * IMAGE_H
    cv2.imwrite(os.path.join(args.out, 'registration.png'), panel)
    cv2.imwrite(os.path.join(args.out, 'checkerboard_after.png'), cb_after)

    print('=== NeuFlow v3 sparse-query registration ===')
    print(f'queried points         : {n_query}  ({100.0*n_query/dense_px:.2f}% of {dense_px}-px dense flow)')
    print(f'RANSAC inlier ratio    : {inlier_ratio*100:.1f}%')
    print(f'homography corner shift: {corner_shift:.2f} px  (near 0 = trivial/identity, so motion is real)')
    print(f'photometric error      : {err_nowarp:.1f} -> {err_warp:.1f}  ({photo_gain:+.0f}% vs no warp)')
    print(f'reproj error (inliers) : {reproj_err:.2f} px  [self-consistency, NOT accuracy vs GT]')
    print(f'saved                  : {os.path.join(args.out, "registration.png")}')


if __name__ == '__main__':
    main()
