#!/bin/bash
# One-shot Explorer setup for NeuFlow v3. Idempotent — safe to re-run.
# Usage (in an OOD "Explorer Shell" or ssh session):
#   git clone https://github.com/shrirag10/Neuflowv3.git ~/NeuFlow_v3 2>/dev/null
#   cd ~/NeuFlow_v3 && git checkout v3-rebuild && bash hpc/bootstrap.sh
set -e

echo "== 1/4 conda module =="
# module names differ per site config; detect what exists
ANACONDA_MOD=$(module -t avail 2>&1 | grep -iE "^(anaconda|miniconda)" | sed 's/[[:space:]]*<.*//' | sort -V | tail -1)
if [ -z "$ANACONDA_MOD" ]; then
    echo "No anaconda/miniconda module found. Run 'module avail' and load manually."
    exit 1
fi
echo "using module: $ANACONDA_MOD"
module load "$ANACONDA_MOD"

echo "== 2/4 conda env (neuflow) =="
if ! conda env list | grep -q "^neuflow "; then
    conda create -y -n neuflow python=3.10
fi
source activate neuflow 2>/dev/null || conda activate neuflow
# torch pip wheels bundle the CUDA runtime; only the node's driver matters
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet opencv-python-headless matplotlib tqdm huggingface_hub h5py

echo "== 3/4 scratch + dataset symlink =="
mkdir -p /scratch/$USER/neuflow_datasets /scratch/$USER/neuflow_ckpts
if [ ! -e datasets ]; then ln -s /scratch/$USER/neuflow_datasets datasets; fi

echo "== 4/4 sanity =="
python3 - <<'PY'
import torch
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
else:
    print('(login nodes have no GPU — this is expected; GPU check happens in the job)')
PY
echo
echo "READY. Next:"
echo "  1) upload datasets+weights from your laptop (see hpc/explorer_setup.md, ~83 GB)"
echo "  2) verify GPU quickly:  srun --partition=gpu-interactive --gres=gpu:v100-sxm2:1 --mem=8G --time=00:10:00 --pty python3 -c 'import torch; print(torch.cuda.get_device_name(0))'"
echo "  3) submit:              sbatch hpc/train_big18.sbatch"
