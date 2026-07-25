#!/bin/bash
#SBATCH --job-name=nf3-distill3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log

# absolute env interpreter: immune to module/activation quirks in batch shells
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/neuflow/bin/python3
$PY -c "import torch" || { echo "ENV BROKEN: $PY lacks torch" >&2; exit 1; }
cd $HOME/NeuFlow_v3

CKPT_DIR=/scratch/$USER/neuflow_ckpts/distill3
mkdir -p "$CKPT_DIR"


$PY scripts/train_distill.py \
  --stage vkitti2_mix \
  --batch_size 16 --num_workers 8 \
  --lr 1e-4 --num_steps 60000 --student_iters 3 \
  --checkpoint_dir "$CKPT_DIR"
