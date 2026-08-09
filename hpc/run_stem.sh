#!/bin/bash
# Submit the full-resolution stem experiment. Run ON the cluster:
#
#     cd ~/NeuFlow_v3 && ./hpc/run_stem.sh
#
# Two jobs, not one. The stem alone tests the diagnosis that the decoder's
# 1/8-resolution input is what caps sub-pixel accuracy. The stem+PE pair tests
# whether the Fourier encoding's earlier null flips now that there is
# full-resolution content for it to index into. Same cost in parallel, different
# questions.
#
# Everything else is identical to v3_FlyingChairs_VKITTI2_Sintel, which is the
# run to compare against: 2.392 EPE, 75.83% 1px.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> repo at $(git log --oneline -1)"
echo

for f in hpc/v3_FlyingChairs_VKITTI2_Sintel_stem.sbatch \
         hpc/v3_FlyingChairs_VKITTI2_Sintel_stem_pe.sbatch; do
    [ -f "$f" ] || { echo "missing $f -- is the repo synced?" >&2; exit 1; }
    sbatch "$f"
done

echo
squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %.10l %R"

cat <<'EOF'

Watch:    squeue -u $USER
Logs:     tail -n 5 ~/NeuFlow_v3/nf3-stem*.log
Evaluate: sbatch hpc/eval_stem.sbatch        (once both are COMPLETED)
EOF
