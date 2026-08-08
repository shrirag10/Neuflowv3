#!/bin/bash
#SBATCH --job-name=nf3-tartanair-dl
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/dl_logs/tartanair_%j.log

# Two environments only, not the whole 3 TB release. download_training.py has no
# environment filter, so we address the bucket directly: anonymous S3-compatible
# reads, keys of the form <env>/Easy/<part>.zip.
#
#   amusement   22.4 GB   outdoor
#   carwelding  10.4 GB   indoor industrial
#
# Both are drone trajectories, which is the point: the margin rule was derived on
# a car at 26.6 px mean motion and needs testing at a different motion scale.

set -e
export PYTHONNOUSERSITE=1
PY=$HOME/.conda/envs/neuflow/bin/python3
DEST=/scratch/$USER/neuflow_datasets/tartanair
mkdir -p "$DEST" /scratch/$USER/dl_logs
cd "$DEST"

# The AirLab bucket is on port 8080, which this cluster's squid proxy rejects
# (403), and the Cloudflare R2 mirror needs signed S3 requests (400). The
# HuggingFace mirror is plain HTTPS on 443 and works: verified 200 with the
# exact manifest size. curl -L follows HF's redirect to its CDN; -C - resumes.
BASE=https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main

for env in amusement carwelding; do
  for part in image_left flow_flow flow_mask; do
    out="${env}_Easy_${part}.zip"
    if [ -s "$out" ]; then echo "exists $out"; continue; fi
    echo "downloading $env/Easy/$part.zip"
    curl -fL -C - --retry 5 --retry-delay 10 -o "$out" \
         "$BASE/$env/Easy/$part.zip" || { echo "FAILED $out" >&2; exit 1; }
    echo "  done $out $(du -h "$out" | cut -f1)"
  done
done
echo DOWNLOAD_DONE

for z in *.zip; do
  echo "unzip $z"
  unzip -qn "$z" -d .
done

echo "layout:"
find . -maxdepth 4 -type d | head -12
echo "flow files: $(find . -name '*_flow.npy' | wc -l)"
echo "mask files: $(find . -name '*_mask.npy' | wc -l)"
echo "images:     $(find . -name '*_left.png' | wc -l)"
echo TARTANAIR_READY
