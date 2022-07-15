#!/bin/bash

PYTHON_EXE=/home/local/VANDERBILT/xuk9/anaconda3/envs/DeepCAC/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/DeepCAC

#CUDA_VISIBLE_DEVICES=1 ${PYTHON_EXE} \
#  ${SRC_ROOT}/src/run_step1_heart_localization.py \
#  --conf ${SRC_ROOT}/MASI_utils/eval_VLSP/config_pilot/step1_heart_localization.yaml

#CUDA_VISIBLE_DEVICES=1 ${PYTHON_EXE} \
#  ${SRC_ROOT}/src/run_step2_heart_segmentation.py \
#  --conf ${SRC_ROOT}/MASI_utils/eval_VLSP/config_pilot/step2_heart_segmentation.yaml

#CUDA_VISIBLE_DEVICES=1 ${PYTHON_EXE} \
#  ${SRC_ROOT}/src/run_step3_cac_segmentation.py \
#  --conf ${SRC_ROOT}/MASI_utils/eval_VLSP/config_pilot/step3_cac_segmentation.yaml

CUDA_VISIBLE_DEVICES=1 ${PYTHON_EXE} \
  ${SRC_ROOT}/src/run_step4_cac_scoring.py \
  --conf ${SRC_ROOT}/MASI_utils/eval_VLSP/config_pilot/step4_cac_scoring.yaml