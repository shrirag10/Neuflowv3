#!/bin/bash
# =============================================================================
#  NeuFlow v3 — BASELINE v1 (paper-aligned base parameters)
#
#  Every parameter below is traceable to a source paper — see
#  docs/base_parameters.md for the full derivation table.
#
#  Sources:
#    [V2]  NeuFlow-V2 (arXiv:2408.10161) — trained w/ RAFT recipe, iters 1+8
#    [AF]  AnyFlow (CVPR 2023)           — RAFT loss gamma, one-cycle, AdamW
#    [ID]  InfiniDepth                   — N random coord-value pairs, L1 loss
# =============================================================================

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/neuflowv3_baseline_v1"
PRETRAINED="neuflow_mixed.pth"          # v2 weights, backbone frozen

BATCH_SIZE=4          # VRAM-bound on RTX 4060 8GB
CROP_H=256            # (dataset stage vkitti2: crop set in datasets.py)
SPARSE_N=4096         # [ID] random coordinate-value pairs per image (3.1% of crop)
ADAPTIVE_RATIO=0.5    # 50% at motion boundaries, 50% uniform
LR=2e-4               # decoder-only; one-cycle peak [AF]
NUM_STEPS=30000
MAX_FLOW=400

mkdir -p "${CKPT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --stage                vkitti2 \
  --implicit \
  --sparse_loss \
  --num_sparse_points    ${SPARSE_N} \
  --adaptive_query_ratio ${ADAPTIVE_RATIO} \
  --no_query_jitter \
  --gamma                0.8 \
  --train_iters_s16      1 \
  --train_iters_s8       8 \
  --onecycle \
  --batch_size           ${BATCH_SIZE} \
  --num_workers          4 \
  --lr                   ${LR} \
  --num_steps            ${NUM_STEPS} \
  --val_freq             2000 \
  --val_dataset          none \
  --max_flow             ${MAX_FLOW} \
  --resume               "${PRETRAINED}" \
  --checkpoint_dir       "${CKPT_DIR}"

echo "Evaluate: python3 scripts/eval_vkitti2.py --checkpoint ${CKPT_DIR}/step_030000.pth"
