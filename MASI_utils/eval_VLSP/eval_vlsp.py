import os
import random
import numpy as np


def random_cohort_pilot():
    n_case = 4
    in_ct_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/VLSP_all/ct_std'
    nii_file_list = os.listdir(in_ct_dir)
    random.shuffle(nii_file_list)
    nii_file_list = nii_file_list[:4]

    out_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/DeepCAC/VLSP_pilot/raw'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for nii_file in nii_file_list:
        case_dir = os.path.join(out_dir, nii_file.replace('.nii.gz', ''))
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        img_path = os.path.join(case_dir, 'img.nii.gz')
        in_nii = os.path.join(in_ct_dir, nii_file)
        ln_cmd = 'ln -sf %s %s' % (in_nii, img_path)
        os.system(ln_cmd)


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


if __name__ == '__main__':
    # random_cohort_pilot()
    prepare_vlsp_in_chunk()