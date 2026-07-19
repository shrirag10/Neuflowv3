#!/bin/bash
# Spring train split, left camera, forward flow only.
# Uses the official cv-stuttgart/spring_utils downloader (DaRUS dataverse).
# SIZE WARNING: frames ~40 GB + flo5 GT ~250 GB. Scratch only. Takes hours.
set -e
DEST=/scratch/$USER/neuflow_datasets
mkdir -p $DEST && cd $DEST
[ -d spring_utils ] || git clone https://github.com/cv-stuttgart/spring_utils.git
cd spring_utils
python3 download_spring.py --split train --data frame_left flow_FW_left --dest $DEST/spring
echo "Spring ready under $DEST/spring/train (check: ls $DEST/spring/train | head)"
