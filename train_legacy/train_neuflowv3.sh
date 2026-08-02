#!/bin/bash
# =============================================================================
#  NeuFlow v3 — Training Script (Frozen Backbone, 3-Scale Decoder)
#
#  Why FROZEN backbone is correct for our budget:
#    - InfiniDepth trains 800K steps to stabilize end-to-end joint training.
#    - With only 30K steps, updating backbone + decoder simultaneously causes
#      a non-stationary target: backbone features shift every step, decoder
#      chases a moving target → divergence (observed: EPE oscillates 1.9–54).
#    - Frozen backbone = STABLE features = STABLE learning target.
#    - Decoder learns the residual on top of already-excellent coarse flow.
#    - This is standard "linear probing → fine-tuning" protocol in transfer
#      learning, but we skip phase 2 because our budget doesn't support it.
#
#  3-Scale decoder (InfiniDepth Eq. 3, applied twice):
#    h1 = context_s8   (64d)  — appearance / refinement context
#    h2 = FFN1( feat_s8  + g1 ⊙ Linear(h1) )   (128d)
#    h3 = FFN2( feat_s16 + g2 ⊙ Linear(h2) )   (128d)
#
#  Improvements over the run that got EPE=3.20:
#    - 3-scale decoder (was 2-scale before, EPE=3.20)
#    - Zero-init always applied regardless of mode (bug was fixed)
#    - 30K steps (was 20K before)
#    - LR warmup: ramp 0→2e-4 over first 1K steps for smooth start
# =============================================================================

set -e
cd "$(dirname "$0")"

CKPT_DIR="checkpoints/neuflowv3"
PRETRAINED="neuflow_mixed.pth"

BATCH_SIZE=4
NUM_WORKERS=4
SPARSE_N=4096
ADAPTIVE_RATIO=0.5
LR=2e-4         # Decoder only — backbone is FROZEN
NUM_STEPS=30000
MAX_FLOW=400

echo "========================================================"
echo " NeuFlow v3 — Frozen Backbone Training"
echo "   Backbone    : FROZEN (features stable)"
echo "   Decoder  lr : ${LR}"
echo "   Steps       : ${NUM_STEPS}"
echo "   Sparse N    : ${SPARSE_N}"
echo "   3 Scales    : ctx_s8(64d) + feat_s8(128d) + feat_s16(128d)"
echo "========================================================"

rm -rf "${CKPT_DIR}" && mkdir -p "${CKPT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
  --stage              vkitti2          \
  --implicit                            \
  --sparse_loss                         \
  --num_sparse_points  ${SPARSE_N}     \
  --adaptive_query_ratio ${ADAPTIVE_RATIO} \
  --batch_size         ${BATCH_SIZE}   \
  --num_workers        ${NUM_WORKERS}  \
  --lr                 ${LR}           \
  --num_steps          ${NUM_STEPS}    \
  --val_freq           5000            \
  --val_dataset        none            \
  --max_flow           ${MAX_FLOW}     \
  --resume             "${PRETRAINED}" \
  --checkpoint_dir     "${CKPT_DIR}"

echo ""
echo "========================================================"
echo " Training complete."
echo " Checkpoint: ${CKPT_DIR}/step_030000.pth"
echo ""
echo " Evaluate:"
echo "   python3 eval_vkitti2.py \\"
echo "     --checkpoint ${CKPT_DIR}/step_030000.pth \\"
echo "     --dataset_root datasets/vkitti2 --val_scenes Scene18 Scene20"
echo "========================================================"
