"""
  ----------------------------------------
    HeartSeg - run DeepCAC pipeline step2
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
  Deep-learning-based heart segmentation in
  Chest CT scans - all param.s are parsed
  from a config file stored under "/config"
  
"""

import os
import yaml
import argparse
import matplotlib      
matplotlib.use('Agg')

from step2_heartseg import compute_bbox
from step2_heartseg import crop_data
from step2_heartseg import input_data_prep
from step2_heartseg import run_inference
from step2_heartseg import upsample_results
from step2_heartseg import compute_metrics

## ----------------------------------------

# base_conf_file_path = 'config/'
# conf_file_list = [f for f in os.listdir(base_conf_file_path) if f.split('.')[-1] == 'yaml']


def run_process(conf_dict):
    # input-output
    data_folder_path = os.path.normpath(conf_dict["io"]["path_to_data_folder"])
    
    heartloc_data_folder_name = conf_dict["io"]["heartloc_data_folder_name"]
    heartseg_data_folder_name = conf_dict["io"]["heartseg_data_folder_name"]
    
    curated_data_folder_name = conf_dict["io"]["curated_data_folder_name"]
    step1_inferred_data_folder_name = conf_dict["io"]["step1_inferred_data_folder_name"]
    
    bbox_folder_name = conf_dict["io"]["bbox_folder_name"]
    cropped_data_folder_name = conf_dict["io"]["cropped_data_folder_name"]
    model_input_folder_name = conf_dict["io"]["model_input_folder_name"]
    # model_weights_folder_name = conf_dict["io"]["model_weights_folder_name"]
    model_output_folder_name = conf_dict["io"]["model_output_folder_name"]
    upsampled_data_folder_name = conf_dict["io"]["upsampled_data_folder_name"]
    seg_metrics_folder_name = conf_dict["io"]["seg_metrics_folder_name"]
    
    # preprocessing and inference parameters
    has_manual_seg = conf_dict["processing"]["has_manual_seg"]
    fill_mask_holes = conf_dict["processing"]["fill_mask_holes"]
    export_png = conf_dict["processing"]["export_png"]
    num_cores = conf_dict["processing"]["num_cores"]
    use_inferred_masks = conf_dict["processing"]["use_inferred_masks"]
    curated_size = conf_dict["processing"]["curated_size"]
    curated_spacing = conf_dict["processing"]["curated_spacing"]
    inter_size = conf_dict["processing"]["inter_size"]
    training_size = conf_dict["processing"]["training_size"]
    final_size = conf_dict["processing"]["final_size"]
    final_spacing = conf_dict["processing"]["final_spacing"]
    
    # model config
    # weights_file_name = conf_dict["model"]["weights_file_name"]
    down_steps = conf_dict["model"]["down_steps"]
    
    
    if has_manual_seg:
      # if manual segmentation masks are available and "use_inferred_masks" is set to False
      # (in the config file), use the manual segmentation masks to compute the localization 
      run = "Test" if use_inferred_masks else "Train"
    else:
      
      # signal the user if "use_inferred_masks" if run is forced to "Test"
      if not use_inferred_masks:
        print "Manual segmentation masks not provided. Forcing localization with the inferred masks."
      
      # if manual segmentation masks are not available, force "use_inferred_masks" to True
      run = "Test"
      
    ## ----------------------------------------
    
    # set paths: step1 and step2
    heartloc_data_path = os.path.join(data_folder_path, heartloc_data_folder_name)
    heartseg_data_path = os.path.join(data_folder_path, heartseg_data_folder_name)
    if not os.path.exists(heartseg_data_path): os.makedirs(heartseg_data_path)
    
    # set paths: results from step 1 - heart localisation
    curated_dir_path = os.path.join(heartloc_data_path, curated_data_folder_name)
    step1_inferred_dir_path = os.path.join(heartloc_data_path, step1_inferred_data_folder_name)
    
    bbox_dir_path = os.path.join(heartseg_data_path, bbox_folder_name)
    cropped_dir_name = os.path.join(heartseg_data_path, cropped_data_folder_name)
    
    # set paths: model processing
    model_input_dir_path = os.path.join(heartseg_data_path, model_input_folder_name)
    # model_weights_dir_path = os.path.join(heartseg_data_path, model_weights_folder_name)
    model_output_dir_path = os.path.join(heartseg_data_path, model_output_folder_name)
    
    # set paths: final location where the inferred masks (NRRD) and the metrics,
    # computed if the manual segmentation masks are available, will be stored
    model_output_nrrd_dir_path = os.path.join(heartseg_data_path, upsampled_data_folder_name)
    result_metrics_dir_path = os.path.join(heartseg_data_path, seg_metrics_folder_name)
    
    
    # create the subfolders where the results are going to be stored
    if not os.path.exists(bbox_dir_path): os.mkdir(bbox_dir_path)
    if not os.path.exists(cropped_dir_name): os.mkdir(cropped_dir_name)
    if not os.path.exists(model_input_dir_path): os.mkdir(model_input_dir_path)
    
    # assert the curated data folder exists and it is non empty
    assert os.path.exists(curated_dir_path)
    assert len(os.listdir(curated_dir_path))
    
    # assert the inference data folder exists and it is non empty
    assert os.path.exists(step1_inferred_dir_path)
    assert len(os.listdir(step1_inferred_dir_path))
    
    
    # assert the weights folder exists and the weights file is found
    # weights_file = os.path.join(model_weights_dir_path, weights_file_name)
    # assert os.path.exists(weights_file)
    
    if not os.path.exists(model_output_dir_path): os.mkdir(model_output_dir_path)
    if not os.path.exists(model_output_nrrd_dir_path): os.mkdir(model_output_nrrd_dir_path)
    if not os.path.exists(result_metrics_dir_path): os.mkdir(result_metrics_dir_path)
    
    ## ----------------------------------------
    
    # run the segmentation pipeline
    print "\n--- STEP 2 - HEART SEGMENTATION ---\n"
    
    #
    compute_bbox.compute_bbox(cur_dir = curated_dir_path,
                              pred_dir = step1_inferred_dir_path,
                              output_dir = bbox_dir_path,
                              num_cores = num_cores,
                              has_manual_seg = has_manual_seg,
                              run = run)
    
    #
    crop_data.crop_data(bb_calc_dir = bbox_dir_path,
                        output_dir = cropped_dir_name,
                        network_dir = model_input_dir_path,
                        inter_size = inter_size,
                        final_size = final_size,
                        final_spacing = final_spacing,
                        num_cores = num_cores)
    
    #
    input_data_prep.input_data_prep(input_dir = cropped_dir_name,
                                    output_dir = model_input_dir_path,
                                    run = run,
                                    fill_holes = fill_mask_holes,
                                    final_size = final_size)
    
    #
    run_inference.run_inference(data_dir = model_input_dir_path,
                                output_dir = model_output_dir_path,
                                export_png = export_png,
                                final_size = final_size,
                                training_size = training_size,
                                down_steps = down_steps,
                                model_weight_path= conf_dict["model"]["model_check_point"]
                                )
    
    #
    upsample_results.upsample_results(cur_input = curated_dir_path,
                                      crop_input = cropped_dir_name,
                                      network_dir = model_input_dir_path,
                                      test_dir = model_output_dir_path,
                                      output_dir = model_output_nrrd_dir_path,
                                      inter_size = inter_size,
                                      num_cores = num_cores)
    
    
    if has_manual_seg == True:
      compute_metrics.compute_metrics(cur_dir = curated_dir_path,
                                      pred_dir = model_output_nrrd_dir_path,
                                      output_dir = result_metrics_dir_path,
                                      raw_spacing = curated_spacing,
                                      num_cores = num_cores,
                                      mask = has_manual_seg)
    else:
      pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pipeline step 2 - heart segmentation.')

    parser.add_argument('--conf',
                        required=False,
                        help='Specify the YAML configuration file containing the run details.' \
                             + 'Defaults to "heart_segmentation.yaml"',
                        # choices = conf_file_list,
                        default="step2_heart_segmentation.yaml",
                        )

    args = parser.parse_args()

    # conf_file_path = os.path.join(base_conf_file_path, args.conf)
    conf_file_path = args.conf

    with open(conf_file_path) as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

        run_process(yaml_conf)