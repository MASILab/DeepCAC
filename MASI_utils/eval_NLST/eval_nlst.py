import functools
import os
import random
import numpy as np
from tqdm import tqdm
import pandas as pd


def prepare_nlst_in_chunk():
    n_chunk = 70000 / 50

    in_ct_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/NLST_all/ct_std'
    nii_file_list = os.listdir(in_ct_dir)
    random.shuffle(nii_file_list)

    full_list = list(range(len(nii_file_list)))
    chunk_list = np.array_split(full_list, n_chunk)

    out_root = '/nfs/masi/xuk9/Projects/DeepCAC/NLST_full'
    if not os.path.exists(out_root): os.makedirs(out_root)
    for chunk_idx, chunk_idx_list in tqdm(enumerate(chunk_list), total=len(chunk_list)):
        chunk_idx_list = chunk_idx_list.tolist()
        # print('Process %d th chunk' % chunk_idx)
        # print('Number of subject: %d' % len(chunk_idx_list))
        chunk_dir = os.path.join(out_root, 'chunk_{:04d}'.format(chunk_idx))
        if not os.path.exists(chunk_dir): os.makedirs(chunk_dir)
        file_name_list = [nii_file_list[idx] for idx in chunk_idx_list]
        raw_dir = os.path.join(chunk_dir, 'raw')
        for file_name in file_name_list:
            case_dir = os.path.join(raw_dir, file_name.replace('.nii.gz', ''))
            if not os.path.exists(case_dir): os.makedirs(case_dir)
            in_ct = os.path.join(in_ct_dir, file_name)
            out_ct = os.path.join(case_dir, 'img.nii.gz')
            ln_cmd = 'ln -sf %s %s' % (in_ct, out_ct)
            os.system(ln_cmd)


def generate_bash_script_for_batch_process():
    n_batch = 4
    script_dir = '/nfs/masi/xuk9/Projects/DeepCAC/NLST_full_run_script'
    if not os.path.exists(script_dir): os.makedirs(script_dir)

    chunk_dir = '/nfs/masi/xuk9/Projects/DeepCAC/NLST_full'
    chunk_list = os.listdir(chunk_dir)

    chunk_batch_list = np.array_split(chunk_list, n_batch)

    for idx_batch, chunk_list in enumerate(chunk_batch_list):
        batch_sh_path = os.path.join(script_dir, 'batch_{:02d}.sh'.format(idx_batch))

        print('Save to {:s}'.format(batch_sh_path))
        with open(batch_sh_path, 'w') as file:
            file.write('#!/bin/bash\n\n')
            file.write('PYTHON_EXE=/home/local/VANDERBILT/xuk9/anaconda3/envs/DeepCAC/bin/python\n')
            file.write('SRC_ROOT=/nfs/masi/xuk9/src/DeepCAC\n')
            file.write('DATA_ROOT=/nfs/masi/xuk9/Projects/DeepCAC/NLST_full\n')
            file.write('\n')
            file.write('CHUNK_LIST=(')
            for chunk_name in chunk_list:
                file.write(' {:s}'.format(chunk_name))
            file.write(' )\n')

            file.write('for CHUNK in \"${CHUNK_LIST[@]}\"\n')
            file.write('do\n')
            file.write('  CUDA_VISIBLE_DEVICES=0 ${PYTHON_EXE} ${SRC_ROOT}/run_chunk.py --chunk-root ${DATA_ROOT}/${CHUNK}\n')
            file.write('done\n')
        chmod_cmd = 'chmod +x {:s}'.format(batch_sh_path)
        os.system(chmod_cmd)


def check_percent_completion():
    chunk_dir = '/nfs/masi/xuk9/Projects/DeepCAC/NLST_full'
    chunk_list = os.listdir(chunk_dir)

    n_w_result_csv = 0
    for chunk in chunk_list:
        result_csv_path = os.path.join(chunk_dir, chunk, 'step4_cac_score', 'cac_score_results.csv')
        if os.path.exists(result_csv_path):
            n_w_result_csv += 1

    print('{:d} completed {:.1%}'.format(n_w_result_csv, float(n_w_result_csv) / float(len(chunk_list))))


if __name__ == '__main__':
    # prepare_nlst_in_chunk()
    # generate_bash_script_for_batch_process()
    check_percent_completion()
