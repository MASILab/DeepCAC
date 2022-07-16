import os
import random
from args import get_default_conf_dict
from src.run_step1_heart_localization import run_process as run_step1
from src.run_step2_heart_segmentation import run_process as run_step2
from src.run_step3_cac_segmentation import run_process as run_step3
from src.run_step4_cac_scoring import run_process as run_step4
import numpy as np


def eval_vlsp_pilot():
    proj_data_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/DeepCAC/VLSP_pilot'

    conf_dict = {}
    step_list = ['step{:d}'.format(step_idx) for step_idx in range(1, 5)]
    for step in step_list:
        step_conf = get_default_conf_dict(step)
        step_conf['io']['path_to_data_folder'] = proj_data_dir
        conf_dict[step] = step_conf

    run_step1(conf_dict['step1'])
    run_step2(conf_dict['step2'])
    run_step3(conf_dict['step3'])
    run_step4(conf_dict['step4'])


def prepare_vlsp_in_chunk():
    n_chunk = 5

    in_ct_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/VLSP_all/ct_std'
    nii_file_list = os.listdir(in_ct_dir)
    random.shuffle(nii_file_list)

    full_list = list(range(len(nii_file_list)))
    chunk_list = np.array_split(full_list, n_chunk)

    out_root = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/DeepCAC/VLSP_full'
    if not os.path.exists(out_root): os.makedirs(out_root)
    for chunk_idx, chunk_idx_list in enumerate(chunk_list):
        chunk_idx_list = chunk_idx_list.tolist()
        print('Process %d th chunk' % chunk_idx)
        print('Number of subject: %d' % len(chunk_idx_list))
        chunk_dir = os.path.join(out_root, 'chunk_{:02d}'.format(chunk_idx))
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


def eval_vlsp_full():
    proj_data_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/DeepCAC/VLSP_full'
    n_chunk = 5
    n_proc = 3
    step_list = ['step{:d}'.format(step_idx) for step_idx in range(1, 5)]
    for chunk_idx in range(n_chunk):
        print()
        print('########################################')
        print('Process chunk_{:02d}'.format(chunk_idx))
        print()
        chunk_dir = os.path.join(proj_data_dir, 'chunk_{:02d}'.format(chunk_idx))
        conf_dict = {}
        for step in step_list:
            step_conf = get_default_conf_dict(step)
            step_conf['io']['path_to_data_folder'] = chunk_dir
            step_conf['processing']['num_cores'] = n_proc
            conf_dict[step] = step_conf

        run_step1(conf_dict['step1'])
        run_step2(conf_dict['step2'])
        run_step3(conf_dict['step3'])
        run_step4(conf_dict['step4'])


if __name__ == "__main__":
    # eval_vlsp_pilot()
    eval_vlsp_full()
