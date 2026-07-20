#!/bin/bash
#SBATCH --job-name=nf3-unfrozen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log

module load $(module -t avail 2>&1 | grep -iE "^(anaconda|miniconda)" | sed 's/[[:space:]]*<.*//' | sort -V | tail -1)
source activate neuflow
cd $HOME/NeuFlow_v3

CKPT_DIR=/scratch/$USER/neuflow_ckpts/unfrozen
mkdir -p "$CKPT_DIR"

# Backbone unfrozen at 1e-5 (100x below decoder LR). UNVERIFIED for this model:
# diverged at batch 4 on laptop; batch 16 is the standard fix, not a guarantee.
python3 train.py \
  --stage vkitti2_mix \
  --implicit --sparse_loss --unfreeze_all --lr_backbone 1e-5 \
  --num_sparse_points 4096 --adaptive_query_ratio 0.5 --no_query_jitter \
  --batch_size 16 --num_workers 8 \
  --lr 2e-4 --onecycle --gamma 0.8 \
  --train_iters_s16 2 --train_iters_s8 4 \
  --num_steps 100000 --val_freq 5000 --val_dataset none \
  --max_flow 400 \
  --resume checkpoints/neuflowv3_rebuild/step_030000.pth --no_zero_init_decoder_head \
  --checkpoint_dir "$CKPT_DIR"
