# -*- coding: utf-8 -*-
"""
    Program: Cardiac_PC

    A python wrapped deep learning library for the medical imaging analysis
        to include the
            1) pre-processing preparation for nets
            2) deep learning nets
            3) post-processing connecting nets to medical imaging

    file name: script_crop_lv_volume.py
    date of creation: 2019-09-29
    date of modification: 2019-09-29
    Author: Meng  Ye
    purpose:

"""
# depent libraries imports
import SimpleITK as sitk
import numpy as np
import os,csv,json
from scipy.io import savemat
import scipy.io as scio
from skimage.measure import label
import matplotlib.pyplot as plt

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

# find region corner index
def find_corner_index(line, threshold):
    _l = len(line)
    down = 0
    up = _l - 1
    for i in range(_l):
        if line[i] > threshold:
            if line[i+1] > threshold:
                down = i
                break
    for i in range(_l -1, -1, -1):
        if line[i] > threshold:
            if line[i-1] > threshold:
                up = i
                break
    return down, up

src_root1 = '/research/cbim/vast/my389/ailab/data/Cardiac super-resolution label maps/smoothed/'
dst_root = '/research/cbim/vast/my389/ailab/data/Cardiac super-resolution label maps/processed'

# if not os.path.exists(dst_root): os.mkdir(dst_root)

for subroot, dirs, files in os.walk(src_root1):
    for file in files:
        if file.endswith('nii.gz') and 'MYO' in file:
            # patient_num = file

            # print('working on', patient_num)

            patient_num_vec = subroot.split(os.path.sep)
            p1 = patient_num_vec[-1]

            if int(p1) < 900: continue
            print('working on: ', p1)

            mask_image = sitk.ReadImage(os.path.join(subroot.replace('smoothed', 'Dataset'), file.replace('_MYO.', '.')))
            origin = mask_image.GetOrigin()
            direction = mask_image.GetDirection()
            spacing = mask_image.GetSpacing()
            size0 = mask_image.GetSize()  # x, y, z
            np_mask_img = sitk.GetArrayFromImage(mask_image)
            np_mask_img = np_mask_img.transpose(1, 0, 2)
            size = np_mask_img.shape

            lv_mask_value = 1
            lv = np.where(np_mask_img == lv_mask_value, lv_mask_value, 0.0)  # z, y, x

            myo_mask_value = 2
            myo = np.where(np_mask_img == myo_mask_value, myo_mask_value, 0.0)  # z, y, x
            myo = myo + 2 * lv
            myo = myo_mask_value - myo

            epi_surface = np.zeros(size)
            endo_surface = np.zeros(size)

            myo_data = scio.loadmat(os.path.join(subroot, file.replace('.nii.gz', '_pc.mat')))['pc']

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(myo_data[:, 0], myo_data[:, 1], myo_data[:, 2], s=3, cmap='viridis')
            # ax.axis()
            # plt.show()

            s = myo_data.shape
            myo_pc = np.zeros([s[0], 4])
            myo_pc[:, 0:3] = myo_data

            # epi = []
            # endo = []

            for k in range(s[0]):
                p = myo_data[k, :]
                px = int(p[0]-1)
                py = int(p[1]-1)
                pz = int(p[2]-1)

                ks = 0
                comparison = True
                while(comparison):
                    if px - ks < 0 or px + ks > size[0] - 1 \
                            or py - ks < 0 or py + ks > size[1] - 1 \
                            or pz - ks < 0 or pz + ks > size[2] - 1:
                        # epi.append(p)
                        epi_surface[px, py, pz] = 1
                        comparison = False
                        continue
                    lv_kernel = lv[px - ks: px + ks + 1, py-ks: py+ks +1, pz - ks:pz + ks + 1]
                    myo_kernel = myo[px - ks: px + ks + 1, py - ks: py + ks + 1, pz - ks:pz + ks + 1]
                    lvsum = np.sum(lv_kernel)
                    myosum = np.sum(myo_kernel)
                    if lvsum < 0.5 and myosum < 0.5:
                        ks = ks + 1
                        continue
                    else:
                        comparison = False
                    if lvsum > myosum:
                        # endo.append(p)
                        endo_surface[px, py, pz] = 1
                        myo_pc[k, 3] = 1
                    else:
                        # epi.append(p)
                        epi_surface[px, py, pz] = 1

            # epi = np.array(epi)
            # endo = np.array(endo)

            # print('epi.shape')
            # print(epi.shape)
            # print('endo.shape')
            # print(endo.shape)
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(myo_pc[:, 0], myo_pc[:, 1], myo_pc[:, 2], s=3, c=myo_pc[:, 3], cmap='viridis')
            # ax.axis()
            # plt.show()
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(epi[:, 0], epi[:, 1], epi[:, 2], s=3, c='b')
            # ax.axis()
            # plt.show()
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(endo[:, 0], endo[:, 1], endo[:, 2], s=3, c='r')
            # ax.axis()
            # plt.show()

            # data_root1 = os.path.join(subroot, file.replace('.nii.gz', '.npz'))
            # np.savez_compressed(data_root1, myo_pc)

            data_root1 = os.path.join(subroot, file.replace('.nii.gz', '_epi_surface.npz'))
            np.savez_compressed(data_root1, epi_surface)

            data_root1 = os.path.join(subroot, file.replace('.nii.gz', '_endo_surface.npz'))
            np.savez_compressed(data_root1, endo_surface)

            data_root1 = os.path.join(subroot, file.replace('.nii.gz', '_epi_surface.mat'))
            scio.savemat(data_root1, {'data': epi_surface})

            data_root1 = os.path.join(subroot, file.replace('.nii.gz', '_endo_surface.mat'))
            scio.savemat(data_root1, {'data': endo_surface})



            # print(12/0)


            # print('finish 1')





