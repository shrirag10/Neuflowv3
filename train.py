import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import os
import csv
from tqdm import tqdm

from data_utils.datasets import build_train_dataset
from NeuFlow.neuflow import NeuFlow
from utils.loss import flow_loss_func, sparse_flow_loss_func
from NeuFlow.adaptive_query import adaptive_flow_query
from data_utils.evaluate import validate_things, validate_sintel, validate_kitti, validate_viper
from utils.load_model import my_load_weights, my_freeze_model
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--stage', default='things', type=str)
    parser.add_argument('--val_dataset', default=['things', 'sintel'], type=str, nargs='+')

    # training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='LR for backbone when using --unfreeze_all (default 1e-5)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--val_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=1000000, type=int)

    parser.add_argument('--max_flow', default=400, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--strict_resume', action='store_true')

    # distributed training
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')

    # model mode
    parser.add_argument('--implicit', action='store_true',
                        help='Use implicit decoder instead of convex upsampler')
    parser.add_argument('--sparse_loss', action='store_true',
                        help='Use sparse-point loss (InfiniDepth-style)')
    parser.add_argument('--num_sparse_points', default=8192, type=int,
                        help='Number of random points for sparse loss')
    parser.add_argument('--adaptive_query_ratio', default=0.5, type=float,
                        help='Fraction of queries at flow-gradient edges (0=uniform, 1=all adaptive)')
    parser.add_argument('--unfreeze_all', action='store_true',
                        help='Unfreeze all parameters (phase 2 end-to-end fine-tuning)')

    return parser


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    print('Use %d GPUs' % torch.cuda.device_count())
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    if args.distributed:
        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist('pytorch', **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure checkpoint directory exists
    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # model
    model = NeuFlow(use_implicit=args.implicit).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    scaler = torch.amp.GradScaler('cuda')

    # Differential learning rates (standard transfer-learning practice):
    #   - Decoder (new, randomly initialised) → high LR to learn quickly
    #   - Backbone (pretrained, specialized)  → low LR to adapt gently
    decoder_params = list(model_without_ddp.implicit_decoder_module.parameters()) \
        if (args.implicit and hasattr(model_without_ddp, 'implicit_decoder_module')) else []
    decoder_ids = {id(p) for p in decoder_params}
    backbone_params = [p for p in model_without_ddp.parameters() if id(p) not in decoder_ids]

    if args.implicit and decoder_params and args.unfreeze_all:
        optimizer = torch.optim.AdamW([
            {'params': decoder_params,  'lr': args.lr,          'name': 'decoder'},
            {'params': backbone_params, 'lr': args.lr_backbone,  'name': 'backbone'},
        ], weight_decay=1e-4)
        print(f'Optimizer: decoder lr={args.lr:.1e}  backbone lr={args.lr_backbone:.1e}')
    else:
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                      weight_decay=1e-4)


    start_step = 0

    if args.resume:

        state_dict = my_load_weights(args.resume)

        model_without_ddp.load_state_dict(state_dict, strict=args.strict_resume)

        # --- InfiniDepth-aligned init ---
        # Always zero-init the decoder output layer regardless of freeze mode.
        # This ensures delta_flow ≈ 0 at step 0, so the backbone sees near-zero
        # gradients from the decoder and isn't corrupted (safe end-to-end start).
        if args.implicit and hasattr(model_without_ddp, 'implicit_decoder_module'):
            out_layer = model_without_ddp.implicit_decoder_module.flow_head.layers[-1]
            torch.nn.init.zeros_(out_layer.weight)
            torch.nn.init.zeros_(out_layer.bias)

        # Freeze everything except decoder (legacy / ablation only).
        # InfiniDepth trains end-to-end; use --unfreeze_all to match that.
        if args.implicit and not args.unfreeze_all:
            my_freeze_model(model)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable params: {trainable} / {sum(p.numel() for p in model.parameters())}')

        if args.checkpoint_dir is not None:
            torch.save({
                'model': model_without_ddp.state_dict()
            }, os.path.join(args.checkpoint_dir, 'step_0.pth'))

    train_dataset = build_train_dataset(args.stage)
    print('Number of training images:', len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, args.lr,
    #     args.num_steps + 10,
    #     pct_start=0.05,
    #     cycle_momentum=False,
    #     anneal_strategy='cos',
    #     last_epoch=last_epoch,
    # )

    total_steps = 0
    epoch = 0

    counter = 0

    pbar = tqdm(total=args.num_steps, desc='Training', unit='step',
                dynamic_ncols=True, initial=total_steps)

    while total_steps < args.num_steps:
        model.train()

        # mannual change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            if total_steps >= args.num_steps:
                break

            optimizer.zero_grad()

            img1, img2, flow_gt, valid = [x.to(device) for x in sample]

            img1 = img1.half()
            img2 = img2.half()

            model_without_ddp.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], device)

            with torch.amp.autocast('cuda', enabled=True):

                # --- Sparse implicit training: sample coords BEFORE forward ---
                if args.implicit and args.sparse_loss:
                    B, _, H, W = flow_gt.shape
                    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
                    valid_mask = (valid >= 0.5) & (mag < args.max_flow)

                    # Adaptive flow-gradient-aware query sampling
                    query_coords = adaptive_flow_query(
                        flow_gt, valid_mask,
                        num_points=args.num_sparse_points,
                        adaptive_ratio=args.adaptive_query_ratio,
                        jitter=True,
                    )  # [B, N, 2]

                    # --- Bilinearly sample GT flow at continuous coords ---
                    # align_corners=False matches the implicit decoder's sampling convention
                    gt_grid = query_coords.clone().float()
                    # align_corners=False pixel-center normalization
                    gt_grid[..., 0] = 2.0 * (gt_grid[..., 0] + 0.5) / W - 1.0
                    gt_grid[..., 1] = 2.0 * (gt_grid[..., 1] + 0.5) / H - 1.0
                    gt_grid.clamp_(-1 + 1e-6, 1 - 1e-6)
                    gt_grid = gt_grid.unsqueeze(1)  # [B, 1, N, 2]  (x,y) order for grid_sample
                    gt_at_query = F.grid_sample(
                        flow_gt, gt_grid, mode='bilinear',
                        padding_mode='border', align_corners=False,
                    ).squeeze(2)  # [B, 2, N]

                    flow_preds = model(img1, img2, iters_s16=4, iters_s8=7,
                                       query_coords=query_coords)

                    # Compute multi-scale loss on sparse predictions
                    loss = torch.tensor(0.0, device=device)
                    n_preds = len(flow_preds)
                    for idx in range(n_preds):
                        i_weight = 0.9 ** (n_preds - idx - 1)
                        pred = flow_preds[idx]
                        if pred.dim() == 4:
                            # Dense coarse predictions: bilinearly sample at query coords
                            pred_sparse = F.grid_sample(
                                pred, gt_grid, mode='bilinear',
                                padding_mode='border', align_corners=False,
                            ).squeeze(2)  # [B, 2, N]
                        else:
                            # Sparse prediction [B, N, 2] → [B, 2, N]
                            pred_sparse = pred.permute(0, 2, 1)

                        i_loss = (pred_sparse - gt_at_query).abs().mean()
                        loss = loss + i_weight * i_loss

                    # Compute EPE on final prediction
                    final_sparse = flow_preds[-1]
                    if final_sparse.dim() == 3:
                        final_sparse = final_sparse.permute(0, 2, 1)  # [B, 2, N]
                    epe = torch.sum((final_sparse - gt_at_query) ** 2, dim=1).sqrt()
                    metrics = {
                        'epe': epe.mean().item(),
                        'mag': mag.mean().item(),
                    }
                else:
                    # --- Dense forward pass (legacy or implicit without sparse) ---
                    flow_preds = model(img1, img2, iters_s16=4, iters_s8=7)
                    loss, metrics = flow_loss_func(flow_preds, flow_gt, valid, args.max_flow)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # GradScaler automatically skips optimizer.step() on inf/nan grads

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)

            scaler.update()

            _dec_lr = optimizer.param_groups[0]['lr']
            _bb_lr  = optimizer.param_groups[-1]['lr'] if len(optimizer.param_groups) > 1 else _dec_lr
            pbar.set_postfix(
                epe=f"{metrics['epe']:.3f}",
                dec_lr=f"{_dec_lr:.1e}",
                bb_lr=f"{_bb_lr:.1e}",
            )
            pbar.update(1)

            total_steps += 1

            # --- Log metrics to CSV for plotting ---
            if args.checkpoint_dir is not None:
                log_path = os.path.join(args.checkpoint_dir, 'train_log.csv')
                write_header = not os.path.exists(log_path)
                _dec_lr = optimizer.param_groups[0]['lr']
                _bb_lr  = optimizer.param_groups[-1]['lr'] if len(optimizer.param_groups) > 1 else _dec_lr
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['step', 'loss', 'epe', 'mag', 'lr_decoder', 'lr_backbone'])
                    writer.writerow([
                        total_steps,
                        f"{loss.item():.6f}",
                        f"{metrics['epe']:.6f}",
                        f"{metrics['mag']:.6f}",
                        f"{_dec_lr:.2e}",
                        f"{_bb_lr:.2e}",
                    ])

            # Always checkpoint+validate on the final step too
            is_final = total_steps >= args.num_steps
            if total_steps % args.val_freq == 0 or is_final:

                if args.local_rank == 0 and args.checkpoint_dir is not None:
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

                val_results = {}

                if 'things' in args.val_dataset:
                    test_results_dict = validate_things(model_without_ddp, device, dstype='frames_cleanpass', validate_subset=True)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'sintel' in args.val_dataset:
                    test_results_dict = validate_sintel(model_without_ddp, device, dstype='final')
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'kitti' in args.val_dataset:
                    test_results_dict = validate_kitti(model_without_ddp, device)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'viper' in args.val_dataset:
                    test_results_dict = validate_viper(model_without_ddp, device)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                if args.local_rank == 0:

                    counter += 1

                    if counter >= 10:

                        for group in optimizer.param_groups:
                            group['lr'] *= 0.8

                        counter = 0

                    # Save validation results when a checkpoint directory is provided
                    if args.checkpoint_dir is not None:
                        val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                        with open(val_file, 'a') as f:
                            f.write('step: %06d lr: %.6f\n' % (total_steps, optimizer.param_groups[-1]['lr']))

                            for k, v in val_results.items():
                                f.write("| %s: %.3f " % (k, v))

                            f.write('\n\n')

                model.train()

            if is_final:
                break

        epoch += 1

    pbar.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
