#!/bin/bash
# Edge-optimized implicit decoder training script
# Note: Ensure you have downloaded VKITTI2 datasets first.

export CUDA_VISIBLE_DEVICES=0

python3 train.py \
  --stage vkitti2 \
  --implicit \
  --sparse_loss \
  --num_sparse_points 8192 \
  --adaptive_query_ratio 0.5 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 2e-4 \
  --num_steps 50000 \
  --val_freq 5000 \
  --val_dataset none \
  --checkpoint_dir checkpoints/implicit_v2_vkitti2
