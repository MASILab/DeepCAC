"""
  ----------------------------------------
     HeartLoc - DeepCAC pipeline step1
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os

import sys
# import tables
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.step1_heartloc.export_data import clip_LAS
from scipy.ndimage import measurements

import heartloc_model

def save_png(patientID, output_dir_png, img, msk, pred):
  maskIndicesMsk = np.where(msk != 0)
  # img = np.swapaxes(img, 0, 2)
  # sag, cor, axi = clip_LAS(img, xyz=(int(img.shape[0]/2), int(img.shape[1]/2), int(img.shape[2]/2)))
  # clip_range = (-1024, 500)
  # if len(clip_range) > 0:
  #   vmin = clip_range[0]
  #   vmax = clip_range[1]
  # else:
  #   vmax = np.max(img)
  #   vmin = np.min(img)
  if len(maskIndicesMsk) == 0:
    trueBB = [np.min(maskIndicesMsk[0]), np.max(maskIndicesMsk[0]),
              np.min(maskIndicesMsk[1]), np.max(maskIndicesMsk[1]),
              np.min(maskIndicesMsk[2]), np.max(maskIndicesMsk[2])]
    cen = [trueBB[0] + (trueBB[1] - trueBB[0]) / 2,
           trueBB[2] + (trueBB[3] - trueBB[2]) / 2,
           trueBB[4] + (trueBB[5] - trueBB[4]) / 2]
  else:
    cen = [int(len(img) / 2), int(len(img) / 2), int(len(img) / 2)]

  #print(pred.shape)
  pred[pred > 0.5] = 1
  pred[pred < 1] = 0
  # pred[pred > 0.5] = 1
  # pred[pred < 0.5] = 0

  # pred = np.swapaxes(pred,0,2)
  # ps,pc,pa = clip_LAS(pred, xyz = (cen[0],cen[1],cen[2]))


  fig, ax = plt.subplots(2, 3, figsize=(32, 16))
  ax[0, 0].imshow(img[cen[0], :, :], cmap='gray')
  ax[0, 1].imshow(img[:, cen[1], :], cmap='gray')
  ax[0, 2].imshow(img[:, :, cen[2]], cmap='gray')

  # ax[0, 0].imshow(sag, cmap='gray')
  # ax[0, 1].imshow(cor, cmap='gray')
  # ax[0, 2].imshow(axi, cmap='gray')
  #
  # ax[0, 0].imshow(msk[cen[0], :, :], cmap='jet', alpha=0.4)
  # ax[0, 1].imshow(msk[:, cen[1], :], cmap='jet', alpha=0.4)
  # ax[0, 2].imshow(msk[:, :, cen[2]], cmap='jet', alpha=0.4)
  #
  # ax[1, 0].imshow(sag, cmap='gray')
  # ax[1, 1].imshow(cor, cmap='gray')
  # ax[1, 2].imshow(axi, cmap='gray')

  ax[1, 0].imshow(pred[cen[0], :, :], cmap='jet', alpha=0.4)
  ax[1, 1].imshow(pred[:, cen[1], :], cmap='jet', alpha=0.4)
  ax[1, 2].imshow(pred[:, :, cen[2]], cmap='jet', alpha=0.4)
  #
  # ax[1, 0].imshow(ps, cmap='jet', alpha=0.4)
  # ax[1, 1].imshow(pc, cmap='jet', alpha=0.4)
  # ax[1, 2].imshow(pa, cmap='jet', alpha=0.4)

  fileName = os.path.join(output_dir_png, patientID + '_' + ".png")
  plt.savefig(fileName)
  plt.close(fig)


def test(model, dataDir, output_dir_npy, output_dir_png, pkl_file,
         test_file, weights_file, mgpu, has_manual_seg, png):
    
  # Get the weight file with the highest weight
  model.load_weights(weights_file)

  # Kaiwen - 20220715
  # Change to use h5py
  # testFileHdf5 = tables.open_file(os.path.join(dataDir, test_file), "r")
  testFileHdf5 = h5py.File(os.path.join(dataDir, test_file), 'r')
  pklData = pickle.load(open(os.path.join(dataDir, pkl_file), 'rb'))

  # Get data in one list for further processing
  testDataRaw = []
  # num_test_imgs = len(testFileHdf5.root.ID)
  num_test_imgs = testFileHdf5['ID'].shape[0]
  for i in range(num_test_imgs):
    # patientID = testFileHdf5.root.ID[i]
    try:
      patientID = testFileHdf5['ID'][i][0]
      # img = testFileHdf5.root.img[i]
      img = testFileHdf5['img'][i]
      if has_manual_seg:
        # msk = testFileHdf5.root.msk[i]
        msk = testFileHdf5['msk'][i]
      else:  # Create empty dummy has_manual_seg with   same size as the image
        sizeImg = len(img)
        msk = np.zeros((sizeImg, sizeImg, sizeImg), dtype=np.float64)
      if not patientID in pklData.keys():
        print('Patient not found in pkl data', patientID)
        continue
      zDif = pklData[patientID][6][2]
      testDataRaw.append([patientID, img, msk, zDif])
    except:
      print('Something wrong with %s, skip the case' % patientID)
  testFileHdf5.close()

  numData = len(testDataRaw)
  size = len(testDataRaw[0][1])
  imgsTrue = np.zeros((numData, size, size, size), dtype=np.float64)
  msksTrue = np.zeros((numData, size, size, size), dtype=np.float64)

  for i in xrange(0, len(testDataRaw) + 1, mgpu):
    imgTest = np.zeros((4, size, size, size), dtype=np.float64)

    for j in range(mgpu):
      # If the number of test images is not mod 4 == 0, just redo the last file severall times
      patientIndex = min(len(testDataRaw) - 1, i + j)
      patientID = testDataRaw[patientIndex][0]
      print 'Processing patient', patientID
      # Store data for score calculation
      imgsTrue[patientIndex, :, :, :] = testDataRaw[patientIndex][1]
      msksTrue[patientIndex, :, :, :] = testDataRaw[patientIndex][2]
      imgTest[j, :, :, :] = testDataRaw[patientIndex][1]

    msksPred = model.predict(imgTest[:, :, :, :, np.newaxis])

    for j in range(mgpu):
      patientIndex = min(len(testDataRaw) - 1, i + j)
      patientID = testDataRaw[patientIndex][0]
      np.save(os.path.join(output_dir_npy, patientID + '_pred'),
              [[patientID], imgsTrue[patientIndex], msksTrue[patientIndex], msksPred[j, :, :, :, 0]])
              #[[patientID], imgsTrue[patientIndex, :, :, :], msksTrue[patientIndex, :, :, :], msksPred[j, :, :, :, 0]])

    if png:
      for j in range(mgpu):
        patientIndex = min(len(testDataRaw) - 1, i + j)
        patientID = testDataRaw[patientIndex][0]
        save_png(patientID, output_dir_png, imgsTrue[patientIndex], msksTrue[patientIndex], msksPred[j, :, :, :, 0])


def run_inference(model_output_dir_path, model_input_dir_path,
                  crop_size, export_png, model_down_steps, extended, has_manual_seg, model_weight_path):
  mgpu = 1

  print "\nDeep Learning model inference using " + str(mgpu) + "xGPUs:"

  output_dir_npy = os.path.join(model_output_dir_path, 'npy')
  output_dir_png = os.path.join(model_output_dir_path, 'png')
  if not os.path.exists(output_dir_npy):
    os.mkdir(output_dir_npy)
  if export_png and not os.path.exists(output_dir_png):
    os.mkdir(output_dir_png)

  test_file = "step1_test_data.h5"
  pkl_file = "step1_downsample_results.pkl"

  weights_file = model_weight_path

  print 'Loading saved model from "%s"'%(weights_file)
  
  input_shape = (crop_size, crop_size, crop_size, 1)
  model = heartloc_model.get_unet_3d(down_steps = model_down_steps,
                                     input_shape = input_shape,
                                     mgpu = mgpu,
                                     ext = extended)

  test(model, model_input_dir_path, output_dir_npy, output_dir_png,
       pkl_file, test_file, weights_file, mgpu, has_manual_seg, export_png)
  # model.summary()
