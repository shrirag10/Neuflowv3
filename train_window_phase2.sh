#!/bin/bash
# Phase 2 local-window fine-tune: train the full implicit decoder, backbone frozen.

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/neuflowv3_window_phase2"
PRETRAINED="checkpoints/neuflowv3_window/step_010000.pth"

BATCH_SIZE=4
NUM_WORKERS=4
SPARSE_N=4096
ADAPTIVE_RATIO=0.5
LR=1e-4
NUM_STEPS=20000
MAX_FLOW=400

mkdir -p "${CKPT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --stage              vkitti2 \
  --implicit \
  --sparse_loss \
  --num_sparse_points  ${SPARSE_N} \
  --adaptive_query_ratio ${ADAPTIVE_RATIO} \
  --batch_size         ${BATCH_SIZE} \
  --num_workers        ${NUM_WORKERS} \
  --lr                 ${LR} \
  --num_steps          ${NUM_STEPS} \
  --val_freq           2000 \
  --val_dataset        none \
  --max_flow           ${MAX_FLOW} \
  --resume             "${PRETRAINED}" \
  --checkpoint_dir     "${CKPT_DIR}" \
  --no_zero_init_decoder_head
