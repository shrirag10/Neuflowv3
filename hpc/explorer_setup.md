# Explorer (Northeastern HPC) — setup for NeuFlow v3

Access: https://ood.explorer.northeastern.edu (SSO; VPN off-campus). OOD gives you
a browser shell (`Clusters > Explorer Shell`), file manager, and job composer.
Everything below also works over plain `ssh <user>@login.explorer.northeastern.edu`.

## One-time setup

```bash
# 1) code
cd $HOME
git clone https://github.com/shrirag10/Neuflowv3.git NeuFlow_v3
cd NeuFlow_v3 && git checkout v3-rebuild

# 2) environment — or just run: bash hpc/bootstrap.sh
# (bootstrap.sh auto-detects the conda module; cuda module not needed for pip wheels)
conda create -y -n neuflow python=3.10
source activate neuflow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python-headless matplotlib tqdm huggingface_hub

# 3) datasets -> scratch (fast parallel FS; home is quota-limited)
mkdir -p /scratch/$USER/neuflow_datasets
# from your laptop:
#   rsync -avP ~/Projects/NEUFLOW/NeuFlow_v3/datasets/vkitti2 <user>@xfer.explorer.northeastern.edu:/scratch/<user>/neuflow_datasets/
#   rsync -avP ~/Projects/NEUFLOW/NeuFlow_v3/datasets/FlyingChairs_release <user>@xfer.explorer.northeastern.edu:/scratch/<user>/neuflow_datasets/
# (use the xfer node, not login; ~83 GB total, expect a few hours from home upload)
ln -s /scratch/$USER/neuflow_datasets datasets   # repo expects ./datasets
```

Checkpoints to upload too (small): `neuflow_mixed.pth` (35 MB) — needed as the frozen
backbone for every decoder training.

## Queue basics

- `sinfo -p gpu` — see GPU nodes. Request with `--gres=gpu:1` and a GPU type
  constraint if needed (e.g. `--constraint=a100` / `v100` depending on availability).
- `squeue -u $USER` — your jobs. `scancel <id>` — kill.
- Time limit: check partition defaults (`scontrol show partition gpu`); the scripts
  below ask for 8 h which fits any of our runs with margin.

## What to run first (in order)

1. `sbatch hpc/train_big18.sbatch` — the direct continuation of local work:
   rebuild recipe, batch 16 instead of 4, 100K steps instead of 30K. This tests the
   two things the laptop could not: batch size and training length.
2. `sbatch hpc/train_unfrozen.sbatch` — unfreeze the backbone at low LR (the
   experiment InfiniDepth needed 8 GPUs for; single A100 handles batch 16 at 384x512).
3. Spring evaluation (data download is ~200 GB for the val subset with GT — do it on
   scratch, script pending until access is confirmed).

Both scripts checkpoint every 5K steps to `/scratch/$USER/neuflow_ckpts/<run>/`;
`rsync` them back or eval on the cluster with `scripts/eval_vkitti2.py`.

## Honest expectations

Nothing here is a result until it runs. Known risks: module names/versions differ
(run `module avail anaconda cuda` and adjust), GPU availability queues, and the
unfrozen run diverged once on the laptop at batch 4 — batch 16 + low backbone LR is
the standard fix but it is unverified for this model.
