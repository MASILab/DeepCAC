import yaml
import SimpleITK as sitk
import h5py
import keras
import numpy as np


def check_input_h5_structure():
    in_h5_path = '/nfs/masi/xuk9/src/DeepCAC/data/step1_heartloc/model_input/step1_test_data.h5'
    db = h5py.File(in_h5_path, 'r')
    print(db['ID'][0][0])
    db.close()


def keras_version():
    print(keras.__version__)


def check_keras_h5():
    in_h5_list = [
        # '/nfs/masi/xuk9/src/DeepCAC/data/step1_heartloc/model_weights/step1_heartloc_model_weights.hdf5.bak',
        '/nfs/masi/xuk9/src/DeepCAC/data/step2_heartseg/model_weights/step2_heartseg_model_weights.hdf5.bak',
        '/nfs/masi/xuk9/src/DeepCAC/data/step3_cacseg/model_weights/step3_cacseg_model_weights.hdf5']
    for h5_path in in_h5_list:
        db = h5py.File(h5_path, 'r')
        print(db.attrs['keras_version'])
        db.close()


if __name__ == '__main__':
    # check_input_h5_structure()
    # keras_version()
    check_keras_h5()
