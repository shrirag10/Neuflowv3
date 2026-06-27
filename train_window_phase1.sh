#!/bin/bash
# Phase 1 local-window fine-tune: train only win_proj_* layers.

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/neuflowv3_window"
PRETRAINED="checkpoints/neuflowv3/step_030000.pth"

BATCH_SIZE=4
NUM_WORKERS=4
SPARSE_N=4096
ADAPTIVE_RATIO=0.5
LR=5e-4
NUM_STEPS=10000
MAX_FLOW=400

mkdir -p "${CKPT_DIR}"

PYTHON_BIN="python3"
if ! python3 -c "import torch" >/dev/null 2>&1; then
  if /usr/bin/python3 -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN="/usr/bin/python3"
    echo "[train_window_phase1] python3 has no torch; falling back to ${PYTHON_BIN}"
  else
    echo "[train_window_phase1] ERROR: torch not found in python3 or /usr/bin/python3" >&2
    exit 1
  fi
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
"${PYTHON_BIN}" train.py \
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
  --window_finetune \
  --no_zero_init_decoder_head
