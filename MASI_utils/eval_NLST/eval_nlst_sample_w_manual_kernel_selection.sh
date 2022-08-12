#!/bin/bash

PYTHON_EXE=/home/xuk9/anaconda3/envs/DeepCAC/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/DeepCAC
#DATA_ROOT=/home/xuk9/Publication/RSNA_Radiology/DeepCAC/w_manual/Data
#DATA_ROOT=/home/xuk9/Publication/RSNA_Radiology/DeepCAC/w_manual_all/Data
#DATA_ROOT=/home/xuk9/Publication/RSNA_Radiology/DeepCAC/sample_5_RAI/Data
#DATA_ROOT=/home/xuk9/Publication/RSNA_Radiology/DeepCAC/sample_5_no_change/Data
#DATA_ROOT=/home/xuk9/Publication/RSNA_Radiology/DeepCAC/w_manual_convert_RAI/Data
#DATA_ROOT=/home/xuk9/Publication/RSNA_Radiology/DeepCAC/w_manual_gaussian/Data
DATA_ROOT=/home/xuk9/Publication/RSNA_Radiology/DeepCAC/NLST_sample_w_manual_plastimatch/Data

CUDA_VISIBLE_DEVICES=0 ${PYTHON_EXE} \
  ${SRC_ROOT}/run_chunk.py \
  --chunk-root ${DATA_ROOT}