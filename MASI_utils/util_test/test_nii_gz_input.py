import SimpleITK as sitk
import tensorflow as tf


def load_vlsp_case():
    in_nii = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/VLSP_all/ct_std/00000001time20131205.nii.gz'
    reader = sitk.ImageFileReader()
    reader.SetFileName(in_nii)
    img_data = reader.Execute()


if __name__ == '__main__':
    load_vlsp_case()