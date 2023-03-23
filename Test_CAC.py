from args import get_default_conf_dict
import os
import argparse
from src.run_step1_heart_localization import run_process as run_step1
from src.run_step2_heart_segmentation import run_process as run_step2
from src.run_step3_cac_segmentation import run_process as run_step3
from src.run_step4_cac_scoring import run_process as run_step4
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # This is setting GPU 0 for the purpose of debugging.
#siemens_path = ['/nfs/Kernel_conversion_outputs/SIE_softtohard/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac/converted','/nfs/Kernel_conversion_outputs/SIE_softtohard/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac/hard', '/nfs/Kernel_conversion_outputs/SIE_softtohard/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac/soft' ]
proj_root = '/nfs/Kernel_conversion_outputs'

all_kernels = [
    {'relative_dir': 'C_D_hardtosoft_2/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac'},
    {'relative_dir': 'C_D_softtohard/train_output/run_4/assessment_single_checkpoint/checkpoint_120/cac'},
    {'relative_dir': 'TOSH_hardtosoft/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac'},
    {'relative_dir': 'TOSH_softtohard/train_output/run_5/assessment_single_checkpoint/checkpoint_120/cac'},
    {'relative_dir': 'GE_BONE_hardtosoft/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac'},
    {'relative_dir': 'GE_BONE_softtohard/train_output/run_1/assessment_single_checkpoint/checkpoint_120/cac'},
    {'relative_dir': 'GE_LUNG_hardtosoft/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac'},
    {'relative_dir': 'GE_LUNG_softtohard/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac'}
]

kerns = ["hard", "soft", "converted"]

if __name__ == "__main__":
    n_proc = 4 #Set to 1 for debugging! original is 4
    conf_dict = {}
    step_list = ['step{:d}'.format(step_idx) for step_idx in range(1, 5)]
    for path in all_kernels:
        for kern in kerns:
            for step in step_list:
                step_conf = get_default_conf_dict(step)
                print("Running CAC for:", os.path.join(proj_root,path['relative_dir'],kern))
                step_conf['io']['path_to_data_folder'] = os.path.join(proj_root, path['relative_dir'],kern)
                step_conf['processing']['num_cores'] = n_proc
                conf_dict[step] = step_conf

            run_step1(conf_dict['step1'])
            run_step2(conf_dict['step2'])
            run_step3(conf_dict['step3'])
            run_step4(conf_dict['step4'])

#'/nfs/Kernel_conversion_outputs/SIE_hardtosoft/train_output/run_0/assessment_single_checkpoint/checkpoint_120/cac/converted': Path for CAC of Siemens hard to soft