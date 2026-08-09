#!/bin/bash
#SBATCH --job-name=nf3-stem-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# Evaluate the stem runs on the cluster, where the dataset already lives, rather
# than pulling 31 MB checkpoints back to a laptop that is 97% full.
#
#     sbatch hpc/eval_stem.sbatch
#     tail -40 nf3-stem-eval_*.log
#
# Runs as a job rather than on the login node, which kills heavy processes.
#
# --stem must match training. It grows the decoder input from 260 to 388, so a
# stem checkpoint will not load into a non-stem model, and the failure is loud.

set -eu
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/neuflow/bin/python3
cd $HOME/NeuFlow_v3

DATA=/scratch/$USER/neuflow_datasets/vkitti2
CKPTS=/scratch/$USER/neuflow_ckpts

for run in v3_FlyingChairs_VKITTI2_Sintel_stem v3_FlyingChairs_VKITTI2_Sintel_stem_pe; do
    CKPT=$CKPTS/$run/step_100000.pth
    if [ ! -f "$CKPT" ]; then
        echo "== $run: no step_100000.pth yet, skipping"
        continue
    fi
    EXTRA=''
    [ "$run" = v3_FlyingChairs_VKITTI2_Sintel_stem_pe ] && EXTRA='--pe'
    echo
    echo "================================================== $run"
    $PY scripts/eval_vkitti2.py \
        --head convex --stem $EXTRA \
        --checkpoint "$CKPT" \
        --dataset_root "$DATA" \
        --fast_dense --stride 2 2>&1 | grep -E 'Val pairs|Mean EPE|1px|3px'
done

cat <<'EOF'

=================================================== reference
same training data, no stem   EPE 2.392   1px 75.83%   3px 88.98%
NeuFlow v2                    EPE 2.324   1px 77.63%   3px 89.80%

1px is the metric the stem targets: the gap lives in sub-pixel precision, not in
large errors. Mean EPE can stay flat while 1px moves, and that is still the
predicted outcome.
EOF
