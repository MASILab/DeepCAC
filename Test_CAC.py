from args import get_default_conf_dict
import os
import argparse
from src.run_step1_heart_localization import run_process as run_step1
from src.run_step2_heart_segmentation import run_process as run_step2
from src.run_step3_cac_segmentation import run_process as run_step3
from src.run_step4_cac_scoring import run_process as run_step4


if __name__ == "__main__":
    n_proc = 4
    conf_dict = {}
    step_list = ['step{:d}'.format(step_idx) for step_idx in range(1, 5)]
    for step in step_list:
        step_conf = get_default_conf_dict(step)
        step_conf['io']['path_to_data_folder'] = '/home-local/DeepCAC/test_data/'
        step_conf['processing']['num_cores'] = n_proc
        conf_dict[step] = step_conf

    run_step1(conf_dict['step1'])
    run_step2(conf_dict['step2'])
    run_step3(conf_dict['step3'])
    run_step4(conf_dict['step4'])
