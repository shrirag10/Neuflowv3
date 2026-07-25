"""Refinement self-distillation (option A).

Goal: make 3 refinement iterations at 1/8 produce the coarse flow that
currently needs 8, by retraining ONLY refine_s8 against the frozen model's
own 8-iteration output. No ground truth involved; the teacher is the model
itself. Everything except refine_s8 stays frozen, so the decoder and all
other v2 weights are untouched.

Student schedule: (1, 3). Teacher schedule: (1, 8).
Loss: L1 between student and teacher coarse flow at 1/8 (valid everywhere).

Usage (HPC):
    python3 scripts/train_distill.py --stage vkitti2_mix --batch_size 16 \
        --lr 1e-4 --num_steps 60000 --student_iters 3 \
        --checkpoint_dir /scratch/$USER/neuflow_ckpts/distill3
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import csv

import torch
from tqdm import tqdm

from data_utils.datasets import build_train_dataset
from NeuFlow.neuflow import NeuFlow
from utils.load_model import my_load_weights, load_with_new_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='vkitti2_mix')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_steps', type=int, default=60000)
    ap.add_argument('--student_iters', type=int, default=3)
    ap.add_argument('--teacher_iters', type=int, default=8)
    ap.add_argument('--resume', default='neuflow_mixed.pth')
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--val_freq', type=int, default=5000)
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model = NeuFlow(use_implicit=True).to(device)
    load_with_new_keys(model, my_load_weights(args.resume),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])

    # teacher: frozen deep copy running the full 8-iteration schedule
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # student: only refine_s8 trains
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith('refine_s8.')
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f'trainable (refine_s8 only): {n_train / 1e6:.2f}M params')

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.num_steps + 10, pct_start=0.05,
        cycle_momentum=False, anneal_strategy='cos')
    scaler = torch.amp.GradScaler('cuda')

    dataset = build_train_dataset(args.stage)
    print('Number of training images:', len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.num_workers,
                                         pin_memory=True, drop_last=True)

    step = 0
    pbar = tqdm(total=args.num_steps, desc='distill', unit='step', dynamic_ncols=True)
    log_path = os.path.join(args.checkpoint_dir, 'train_log.csv')

    while step < args.num_steps:
        for sample in loader:
            if step >= args.num_steps:
                break
            img1, img2 = sample[0].to(device).half(), sample[1].to(device).half()
            model.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], device)
            teacher.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    t_flow = teacher.infer_coarse_state(
                        img1, img2, iters_s16=1, iters_s8=args.teacher_iters)['coarse_flow_s8']
                model.train()
                s_flow = model.infer_coarse_state(
                    img1, img2, iters_s16=1, iters_s8=args.student_iters)['coarse_flow_s8']
                loss = (s_flow.float() - t_flow.float()).abs().mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            pbar.set_postfix(l1=f'{loss.item():.4f}', lr=f'{scheduler.get_last_lr()[0]:.1e}')
            pbar.update(1)

            write_header = not os.path.exists(log_path)
            with open(log_path, 'a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['step', 'l1_vs_teacher', 'lr'])
                w.writerow([step, f'{loss.item():.6f}', f'{scheduler.get_last_lr()[0]:.2e}'])

            if step % args.val_freq == 0 or step >= args.num_steps:
                torch.save({'model': model.state_dict()},
                           os.path.join(args.checkpoint_dir, f'step_{step:06d}.pth'))
    pbar.close()
    print('distillation complete')


if __name__ == '__main__':
    main()
