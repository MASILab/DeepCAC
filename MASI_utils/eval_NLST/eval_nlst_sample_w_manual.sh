#!/bin/bash

PYTHON_EXE=/home/local/VANDERBILT/xuk9/anaconda3/envs/DeepCAC/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/DeepCAC
DATA_ROOT=/nfs/masi/xuk9/Projects/DeepCAC/NLST_full_sample_w_manual

echo CUDA_VISIBLE_DEVICES=0 ${PYTHON_EXE} \
  ${SRC_ROOT}/run_chunk.py \
  --chunk-root ${DATA_ROOT}