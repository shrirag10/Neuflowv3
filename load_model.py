import torch
import re


def my_load_weights(weight_path):

    print('Load checkpoint: %s' % weight_path)

    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(weight_path, map_location=map_location)

    # yolo_checkpoint = torch.load('/media/goku/data/zhiyongzhang/optical_flow/pretrained/yolo_backbone.pth', map_location='cuda')

    state_dict = {}

    for k, v in checkpoint['model'].items():

        # if k.startswith('backbone.block_8_1.'):
        #     continue
        # if k.startswith('backbone.block_cat_8.'):
        #     continue
        # if k.startswith('refine_s16.conv2.'):
        #     continue
        # if k.startswith('refine_s8.conv2.'):
        #     continue
        # if '.running_' in k or '.num_batches' in k:
        #     continue
        # if k.startswith('upserge_s8.'):
        #     continueample_s1.'):
        #     continue

        state_dict[k] = v
        # pass

    # for k, v in yolo_checkpoint['model'].items():
    #     state_dict['backbone.' + k] = v

    return state_dict


def my_freeze_model(model):
    for name, param in model.named_parameters():
        # Works for both single-GPU names (implicit_decoder_module.*)
        # and DDP names (module.implicit_decoder_module.*)
        if 'implicit_decoder_module.' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False