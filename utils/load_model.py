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


def load_with_new_keys(model, state_dict,
                       missing_ok_substrings=(),
                       unexpected_ok_substrings=()):
    """Load a checkpoint while allowing explicitly whitelisted key drift."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    bad_missing = [
        k for k in missing
        if not any(s in k for s in missing_ok_substrings)
    ]
    bad_unexpected = [
        k for k in unexpected
        if not any(s in k for s in unexpected_ok_substrings)
    ]

    if bad_missing:
        raise RuntimeError(f'Missing keys not covered by policy: {bad_missing}')
    if bad_unexpected:
        raise RuntimeError(f'Unexpected checkpoint keys not covered by policy: {bad_unexpected}')

    model.load_state_dict(state_dict, strict=False)
    print(f'Loaded with missing keys initialized fresh: {missing}')
    print(f'Ignored checkpoint-only keys: {unexpected}')


def freeze_for_window_phase1(model):
    for name, param in model.named_parameters():
        param.requires_grad = 'win_proj_' in name
