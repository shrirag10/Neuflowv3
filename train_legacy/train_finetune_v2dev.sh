#!/bin/bash
# =============================================================================
#  NeuFlow v3 — curriculum stage 2: chairs-pretrained decoder -> vkitti2_all
#
#  Warm-starts from the chairs run's final checkpoint. --no_zero_init_decoder_head
#  is REQUIRED: without it the trained convex head is wiped back to bilinear.
# =============================================================================

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/neuflowv3_finetune_v2dev"
PRETRAINED="checkpoints/neuflowv3_chairs_v2dev/step_030000.pth"

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
  --lr                   1e-4 \
  --num_steps            15000 \
  --val_freq             1000 \
  --val_dataset          none \
  --max_flow             400 \
  --resume               "${PRETRAINED}" \
  --no_zero_init_decoder_head \
  --checkpoint_dir       "${CKPT_DIR}"

echo "Evaluate: python3 scripts/eval_vkitti2.py --head convex --checkpoint ${CKPT_DIR}/step_015000.pth"
