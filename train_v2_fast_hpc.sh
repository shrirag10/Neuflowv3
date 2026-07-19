#!/bin/bash
# HPC retraining of NeuFlow v2 at the fast (2,4) iteration schedule.
#
# Finding (2026-07-18, scripts/profile_v2.py): (s16=2, s8=4) runs +38% FPS with
# frozen weights at a 0.13 px EPE cost. Training AT this schedule should recover
# most of that gap (RAFT-family behavior). Target: baseline accuracy at 38 FPS.
#
# Adjust --batch_size and CUDA_VISIBLE_DEVICES to the allocated node.
# Suggested curriculum: chairs (this script) -> things -> vkitti2/mixed finetune.

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/v2_fast24_chairs"
PRETRAINED="neuflow_mixed.pth"   # warm-start from released v2 weights

BATCH_SIZE=16        # HPC GPU: raise to fill VRAM (A100 40GB: 32+)
LR=5e-5          # gentle: warm-starting released weights, only adapting to shorter schedule
NUM_STEPS=100000
MAX_FLOW=400

mkdir -p "${CKPT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train.py \
  --stage              chairs \
  --val_dataset        none \
  --batch_size         ${BATCH_SIZE} \
  --num_workers        8 \
  --lr                 ${LR} \
  --onecycle \
  --gamma              0.8 \
  --train_iters_s16    2 \
  --train_iters_s8     4 \
  --num_steps          ${NUM_STEPS} \
  --val_freq           5000 \
  --max_flow           ${MAX_FLOW} \
  --resume             "${PRETRAINED}" \
  --checkpoint_dir     "${CKPT_DIR}"
# note: v2 (non-implicit) mode trains ALL parameters at --lr; no freeze flags needed

echo "Evaluate: python3 scripts/eval_vkitti2.py --no_implicit --checkpoint ${CKPT_DIR}/step_100000.pth"
echo "(eval runs (1,8) by default; use profile_v2.py sweep with the new checkpoint for (2,4) numbers)"
