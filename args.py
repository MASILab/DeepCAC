def get_default_conf_dict(step):
    if step == 'step1':
        return {
            'io': {
                'path_to_data_folder': '',
                'raw_data_folder_name': 'raw',
                'heartloc_data_folder_name': 'step1_heartloc',
                'curated_data_folder_name': 'curated',
                'qc_curated_data_folder_name': 'curated_qc',
                'resampled_data_folder_name': 'resampled',
                'model_input_folder_name': 'model_input',
                'model_output_folder_name': 'model_output',
                'upsampled_data_folder_name': 'model_output_nrrd',
                'input_scan_postfix': '.nii.gz'
            },
            'processing': {
                'has_manual_seg': False,
                'fill_mask_holes': True,
                'export_png': True,
                'num_cores': 1,
                'create_test_set': 'All',
                'curated_size': [512, 512, 0],
                'curated_spacing': [0.68, 0.68, 2.5],
                'model_input_size': 112,
                'model_input_spacing': 3.0
            },
            'model': {
                'model_check_point': '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/DeepCAC/models/step1_heartloc_model_weights.hdf5',
                'pool_size': [2, 2, 2],
                'conv_size': [3, 3, 3],
                'down_steps': 4,
                'extended': False,
                'training': {}
            }
        }
    elif step == 'step2':
        return {
            'io': {
                'path_to_data_folder': '',
                'raw_data_folder_name': 'raw',
                'heartloc_data_folder_name': 'step1_heartloc',
                'heartseg_data_folder_name': 'step2_heartseg',
                'curated_data_folder_name': 'curated',
                'step1_inferred_data_folder_name': 'model_output_nrrd',
                'bbox_folder_name': 'bbox',
                'cropped_data_folder_name': 'cropped',
                'model_input_folder_name': 'model_input',
                'model_output_folder_name': 'model_output',
                'upsampled_data_folder_name': 'model_output_nrrd',
                'seg_metrics_folder_name': 'model_output_metrics'
            },
            'processing': {
                'has_manual_seg': False,
                'fill_mask_holes': True,
                'export_png': True,
                'num_cores': 1,
                'use_inferred_masks': True,
                'curated_size': [512, 512, 0],
                'curated_spacing': [0.68, 0.68, 2.5],
                'inter_size': [384, 384, 80],
                'training_size': [128, 128, 112],
                'final_size': [128, 128, 80],
                'final_spacing': [0, 0, 2.5]
            },
            'model': {
                'model_check_point': '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/DeepCAC/models/step2_heartseg_model_weights.hdf5',
                'pool_size': [2, 2, 2],
                'conv_size': [3, 3, 3],
                'down_steps': 4,
                'extended': False,
                'training': {}
            }
        }
    elif step == 'step3':
        return {
            'io': {
                'path_to_data_folder': '',
                'raw_data_folder_name': 'raw',
                'heartloc_data_folder_name': 'step1_heartloc',
                'heartseg_data_folder_name': 'step2_heartseg',
                'cacs_data_folder_name': 'step3_cacseg',
                'curated_data_folder_name': 'curated',
                'step2_inferred_data_folder_name': 'model_output_nrrd',
                'dilated_data_folder_name': 'dilated',
                'cropped_data_folder_name': 'cropped',
                'qc_cropped_data_folder_name': 'cropped_qc',
                'model_output_folder_name': 'model_output',
            },
            'processing': {
                'has_manual_seg': False,
                'export_png': True,
                'export_cac_slices_png': True,
                'num_cores': 1,
                'use_inferred_masks': True,
                'patch_size': [32, 48, 48]
            },
            'model': {
                'model_check_point': '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/DeepCAC/models/'
                                     'step3_cacseg_model_weights.hdf5',
            }
        }
    elif step == 'step4':
        return {
            'io': {
                'path_to_data_folder': '',
                'heartloc_data_folder_name': 'step1_heartloc',
                'heartseg_data_folder_name': 'step2_heartseg',
                'cacseg_data_folder_name': 'step3_cacseg',
                'curated_data_folder_name': 'curated',
                'step3_inferred_data_folder_name': 'model_output',
                'cropped_data_folder_name': 'cropped',
                'cac_score_folder_name': 'step4_cac_score'
            },
            'processing': {
                'has_manual_seg': False,
                'num_cores': 1
            }
        }