#!/bin/bash
# =============================================================================
#  NeuFlow v3 — v2-dev: convex-weight head + all-variant training data
#
#  Changes vs train_baseline_v1.sh (see docs/base_parameters.md):
#    --head convex      AnyFlow-style softmax weights over the 3x3 coarse-flow
#                       window + bilinear candidate (bilinear-prior init, starts
#                       at the 2.48-EPE zero-training operating point)
#    --stage vkitti2_all  6 same-trajectory variants = ~6x training pairs,
#                       identical flow GT, appearance-only variation
#    15K steps          overfitting previously set in early; more data may
#                       extend this — watch the eval sweep
# =============================================================================

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/neuflowv3_v2dev"
PRETRAINED="neuflow_mixed.pth"

mkdir -p "${CKPT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --stage                vkitti2_all \
  --implicit \
  --head                 convex \
  --sparse_loss \
  --num_sparse_points    4096 \
  --adaptive_query_ratio 0.5 \
  --no_query_jitter \
  --gamma                0.8 \
  --train_iters_s16      1 \
  --train_iters_s8       8 \
  --onecycle \
  --batch_size           4 \
  --num_workers          4 \
  --lr                   2e-4 \
  --num_steps            15000 \
  --val_freq             1000 \
  --val_dataset          none \
  --max_flow             400 \
  --resume               "${PRETRAINED}" \
  --checkpoint_dir       "${CKPT_DIR}"

echo "Evaluate: python3 scripts/eval_vkitti2.py --head convex --checkpoint ${CKPT_DIR}/step_015000.pth"
