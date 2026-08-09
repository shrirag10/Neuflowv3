#!/bin/bash
#SBATCH --job-name=nf3-stem-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# Evaluate the stem runs against the baseline AT THE SAME STEP.
#
#     sbatch hpc/eval_stem.sbatch          # step 80000, the default
#     STEP=100000 sbatch hpc/eval_stem.sbatch
#
# The stem runs hit the 8 h wall at ~84k steps and last saved at 80k. Comparing
# an 80k stem against a 100k baseline would move two variables at once, and the
# whole point of generating these runs from one template is that they differ in
# exactly one. The baseline kept every 5k checkpoint, so 80k against 80k is
# available and is the honest comparison.
#
# Caveat that belongs on any slide using these numbers: OneCycle had not finished
# annealing at 80k for any of the three, so all are slightly under-trained. That
# affects them equally, which is why the comparison still holds.
#
# Runs as a job rather than on the login node, which kills heavy processes.

set -eu
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/neuflow/bin/python3
cd $HOME/NeuFlow_v3

STEP=${STEP:-080000}
DATA=/scratch/$USER/neuflow_datasets/vkitti2
CKPTS=/scratch/$USER/neuflow_ckpts

echo "Evaluating at step $STEP, VKITTI 2 Scene18+20, fast_dense stride 2"
echo

#      run directory                              extra eval flags
run_eval () {
    local run="$1" extra="$2"
    local ckpt="$CKPTS/$run/step_$STEP.pth"
    if [ ! -f "$ckpt" ]; then
        echo "== $run: no step_$STEP.pth, skipping"
        return
    fi
    echo "================================================== $run  $extra"
    $PY scripts/eval_vkitti2.py --head convex $extra \
        --checkpoint "$ckpt" --dataset_root "$DATA" \
        --fast_dense --stride 2 2>&1 | grep -E 'Mean EPE|1px acc|3px acc'
    echo
}

run_eval v3_FlyingChairs_VKITTI2_Sintel          ''
run_eval v3_FlyingChairs_VKITTI2_Sintel_stem     '--stem'
run_eval v3_FlyingChairs_VKITTI2_Sintel_stem_pe  '--stem --pe'

cat <<EOF
=================================================== how to read this
All three above are at step $STEP, same data, same seed, same schedule. The only
difference is the decoder input, so any gap is the stem.

For context, fully trained at 100k:
  no stem, 100k    EPE 2.392   1px 75.83%   3px 88.98%
  NeuFlow v2       EPE 2.324   1px 77.63%   3px 89.80%

Watch 1px, not mean EPE. The deficit the stem targets is sub-pixel precision, so
1px can move while mean EPE stays flat and that still confirms the diagnosis.
Differences below about 0.05 px EPE are not resolved by a single seed.
EOF
