#!/bin/bash
#SBATCH --job-name=nf3-rebuild-big
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log

module load $(module -t avail 2>&1 | grep -iE "^(anaconda|miniconda)" | sed 's/[[:space:]]*<.*//' | sort -V | tail -1)
source activate neuflow
cd $HOME/NeuFlow_v3

CKPT_DIR=/scratch/$USER/neuflow_ckpts/rebuild_big
mkdir -p "$CKPT_DIR"

python3 train.py \
  --stage vkitti2_mix \
  --implicit --sparse_loss \
  --num_sparse_points 4096 --adaptive_query_ratio 0.5 --no_query_jitter \
  --batch_size 16 --num_workers 8 \
  --lr 2e-4 --onecycle --gamma 0.8 \
  --train_iters_s16 2 --train_iters_s8 4 \
  --num_steps 100000 --val_freq 5000 --val_dataset none \
  --max_flow 400 \
  --resume neuflow_mixed.pth \
  --checkpoint_dir "$CKPT_DIR"
