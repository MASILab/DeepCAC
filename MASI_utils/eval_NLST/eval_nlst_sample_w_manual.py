import functools
import os
import random
import numpy as np
from tqdm import tqdm
import pandas as pd


def archive_data():
    in_csv = '/nfs/masi/xuk9/src/DeepCAC/stats/Results_NLST.csv'
    print('')
    record_df = pd.read_csv(in_csv)

    ct_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/NLST_all/ct_std'
    out_dir = os.path.join(out_root, 'raw')
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    pid_list = record_df['PID'].to_list()

    n_case = len(pid_list)
    n_exist = 0
    for pid in pid_list:
        t0_scan_path = os.path.join(ct_dir, '{:s}time1999.nii.gz'.format(pid))
        if os.path.exists(t0_scan_path):
            n_exist += 1

            out_img_dir = os.path.join(out_dir, pid)
            if not os.path.exists(out_img_dir): os.makedirs(out_img_dir)
            out_img_path = os.path.join(out_img_dir, 'img.nii.gz')

            ln_cmd = 'ln -sf {:s} {:s}'.format(t0_scan_path, out_img_path)
            os.system(ln_cmd)

    print('Total number of cases: {:d}'.format(n_case))
    print('Identified cases with T0 scan {:d}'.format(n_exist))


if __name__ == '__main__':
    out_root = '/nfs/masi/xuk9/Projects/DeepCAC/NLST_full_sample_w_manual'
    if not os.path.exists(out_root): os.makedirs(out_root)

    archive_data()
