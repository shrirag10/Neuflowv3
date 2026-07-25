#!/bin/bash
# Spring train split, left camera, forward flow (verified DaRUS file IDs, 2026-07-19).
# ~61 GB total. Run as a batch job (login nodes kill heavy processes) — see
# /scratch/$USER/spring_dl.sbatch pattern or hpc docs.
set -e
D=/scratch/$USER/neuflow_datasets
cd $D
curl -L -C - -o train_frame_left.zip   "https://darus.uni-stuttgart.de/api/access/datafile/199097"   # 13.4 GB
curl -L -C - -o train_flow_FW_left.zip "https://darus.uni-stuttgart.de/api/access/datafile/199011"   # 47.2 GB
unzip -qn train_frame_left.zip -d spring_tmp
unzip -qn train_flow_FW_left.zip -d spring_tmp
mkdir -p spring
if [ -d spring_tmp/spring ]; then mv spring_tmp/spring/* spring/; else mv spring_tmp/* spring/; fi
echo "done: $(ls spring/train | wc -l) sequences expected under spring/train"
