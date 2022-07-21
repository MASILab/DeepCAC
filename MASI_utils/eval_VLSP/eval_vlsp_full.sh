#!/bin/bash

PYTHON_EXE=/home/local/VANDERBILT/xuk9/anaconda3/envs/DeepCAC/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/DeepCAC
DATA_ROOT=/nfs/masi/xuk9/Projects/DeepCAC/VLSP_full

for VAR in 03 04
do
  echo CUDA_VISIBLE_DEVICES=1 ${PYTHON_EXE} \
    ${SRC_ROOT}/run_chunk.py \
    --chunk-root ${DATA_ROOT}/chunk_${VAR}
done