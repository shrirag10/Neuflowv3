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

$PY - <<'PYEOF'
import os, boto3
from botocore import UNSIGNED
from botocore.client import Config

EP = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
cl = boto3.client("s3", endpoint_url=EP, config=Config(signature_version=UNSIGNED))

envs  = ['amusement', 'carwelding']
parts = ['image_left.zip', 'flow_flow.zip', 'flow_mask.zip']

for env in envs:
    for part in parts:
        key = f'{env}/Easy/{part}'
        out = key.replace('/', '_')
        if os.path.isfile(out):
            print('exists', out, flush=True); continue
        print('downloading', key, flush=True)
        cl.download_file('tartanair', key, out)
        print('  done', out, os.path.getsize(out) / 1e9, 'GB', flush=True)
print('DOWNLOAD_DONE', flush=True)
PYEOF

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
