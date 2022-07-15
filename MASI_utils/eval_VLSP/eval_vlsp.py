import os
import random


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


if __name__ == '__main__':
    random_cohort_pilot()