#!/bin/bash
# =============================================================================
#  NeuFlow v3 — FlyingChairs pretraining with the v2-dev convex head
#
#  Same recipe as train_v2dev.sh but stage=mix_chairs_vkitti2 (~35k pairs, 320x512 crop).
#  Purpose: diverse-data decoder pretraining; evaluate transfer on VKITTI2 and
#  optionally finetune with vkitti2_all afterwards.
# =============================================================================

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/neuflowv3_rebuild"
PRETRAINED="neuflow_mixed.pth"

mkdir -p "${CKPT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --stage                mix_chairs_vkitti2 \
  --implicit \
  --head                 convex \
  --sparse_loss \
  --num_sparse_points    4096 \
  --adaptive_query_ratio 0.5 \
  --no_query_jitter \
  --gamma                0.8 \
  --train_iters_s16      2 \
  --train_iters_s8       4 \
  --onecycle \
  --batch_size           4 \
  --num_workers          4 \
  --lr                   2e-4 \
  --num_steps            30000 \
  --val_freq             2000 \
  --val_dataset          none \
  --max_flow             400 \
  --resume               "${PRETRAINED}" \
  --checkpoint_dir       "${CKPT_DIR}"

echo "Evaluate transfer: python3 scripts/eval_vkitti2.py --head convex --checkpoint ${CKPT_DIR}/step_030000.pth"
