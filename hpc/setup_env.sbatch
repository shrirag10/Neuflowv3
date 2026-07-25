#!/bin/bash
#SBATCH --job-name=nf3-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

# Build the neuflow conda env from zero, as a batch job (login nodes kill
# long pip installs). Idempotent: wipes any half-built env first.
set -e
module load miniconda3/25.9.1

ENVDIR=$HOME/.conda/envs/neuflow
[ -d "$ENVDIR" ] && rm -rf "$ENVDIR"

conda create -y -n neuflow python=3.10
PIP=$ENVDIR/bin/pip
export PYTHONNOUSERSITE=1

$PIP install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
$PIP install --no-cache-dir numpy opencv-python-headless matplotlib tqdm huggingface_hub h5py

$ENVDIR/bin/python3 - <<'EOF'
import torch, cv2, numpy, h5py, tqdm, huggingface_hub
print('torch', torch.__version__, 'cuda build:', torch.version.cuda)
print('ALL_IMPORTS_OK')
EOF
echo ENV_DONE
