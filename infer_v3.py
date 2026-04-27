import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from load_model import my_load_weights
from data_utils import flow_viz


image_width = 512
image_height = 256

checkpoint_path = 'checkpoints/neuflowv3/step_020000.pth'
vis_path = 'results/infer_v3/'


def get_cuda_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image[None].cuda()


def fuse_conv_and_bn(conv, bn):
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels, conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, bias=True,
        ).requires_grad_(False).to(conv.weight.device)
    )
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


device = torch.device('cuda')

model = NeuFlow(use_implicit=True).to(device)
state_dict = my_load_weights(checkpoint_path)
model.load_state_dict(state_dict, strict=True)

for m in model.modules():
    if type(m) is ConvBlock:
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
        delattr(m, 'norm1')
        delattr(m, 'norm2')
        m.forward = m.forward_fuse

model.eval()
model.init_bhwd(1, image_height, image_width, device)

os.makedirs(vis_path, exist_ok=True)

image_path_list = sorted(glob('test_images/*.jpg') + glob('test_images/*.png'))

if len(image_path_list) < 2:
    print('Put at least 2 images in test_images/')
    exit(1)

print(f'Running NeuFlow v3 on {len(image_path_list)-1} pairs → {vis_path}')

for img0_path, img1_path in zip(image_path_list[:-1], image_path_list[1:]):
    image_0 = get_cuda_image(img0_path)
    image_1 = get_cuda_image(img1_path)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        flow = model(image_0, image_1)[-1][0]

    flow_np = flow.permute(1, 2, 0).float().cpu().numpy()
    flow_vis = flow_viz.flow_to_image(flow_np)

    rgb = cv2.resize(cv2.imread(img0_path), (image_width, image_height))
    out = np.vstack([rgb, flow_vis])

    fname = os.path.basename(img0_path)
    cv2.imwrite(os.path.join(vis_path, fname), out)
    print(f'  {fname}  flow mag mean: {np.sqrt((flow_np**2).sum(-1)).mean():.2f} px')

print('Done.')
