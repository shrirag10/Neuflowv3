#!/bin/bash
# MPI-Sintel (~5.3 GB) -> /scratch/$USER/neuflow_datasets/Sintel
set -e
DEST=/scratch/$USER/neuflow_datasets
mkdir -p $DEST && cd $DEST
wget -c http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip -qn MPI-Sintel-complete.zip -d Sintel
rm -f MPI-Sintel-complete.zip
echo "Sintel ready: $(find Sintel/training/flow -name '*.flo' | wc -l) flow files (expect 1041 per pass)"
