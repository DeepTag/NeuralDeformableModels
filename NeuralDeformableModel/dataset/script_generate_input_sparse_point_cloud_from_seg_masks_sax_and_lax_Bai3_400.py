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
import random
import os
import matplotlib.pyplot as plt
from skimage import measure, data
import scipy.io as scio
from scipy.spatial.transform import Rotation as R

# from medinfer.util.logger import get_logger
# logger = get_logger()

def orientation_correction(result_xyz0):
    result_xyz = np.zeros(result_xyz0.shape)
    result_xyz[:, :, 0] = result_xyz0[:, :, 0]
    result_xyz[:, :, 1] = result_xyz0[:, :, 1]
    result_xyz[:, :, 2] = result_xyz0[:, :, 2]
    return result_xyz

def orientation_correction1(result_xyz0):
    result_xyz = np.zeros(result_xyz0.shape)
    result_xyz[:, 0] = result_xyz0[:, 0]
    result_xyz[:, 1] = result_xyz0[:, 2]
    result_xyz[:, 2] = result_xyz0[:, 1]
    return result_xyz

def pc_normalize(data, epi_myo, endo_myo, rv):
    # normalize r_max to 1
    centroid = np.mean(data, axis=0)
    pc = data - centroid
    x_max = np.max(pc[:, 0])
    x_min = np.min(pc[:, 0])

    y_max = np.max(pc[:, 1])
    y_min = np.min(pc[:, 1])

    z_max = np.max(pc[:, 2])
    z_min = np.min(pc[:, 2])

    pc[:, 0] = 1.7 * (pc[:, 0] - x_min) / (x_max - x_min) - 0.85
    pc[:, 1] = 1.7 * (pc[:, 1] - y_min) / (y_max - y_min) - 0.85
    pc[:, 2] = 1.7 * (pc[:, 2] - z_min) / (z_max - z_min) - 0.85

    epi_myo = epi_myo - centroid
    epi_myo[:, 0] = 1.7 * (epi_myo[:, 0] - x_min) / (x_max - x_min) - 0.85
    epi_myo[:, 1] = 1.7 * (epi_myo[:, 1] - y_min) / (y_max - y_min) - 0.85
    epi_myo[:, 2] = 1.7 * (epi_myo[:, 2] - z_min) / (z_max - z_min) - 0.85

    endo_myo = endo_myo - centroid
    endo_myo[:, 0] = 1.7 * (endo_myo[:, 0] - x_min) / (x_max - x_min) - 0.85
    endo_myo[:, 1] = 1.7 * (endo_myo[:, 1] - y_min) / (y_max - y_min) - 0.85
    endo_myo[:, 2] = 1.7 * (endo_myo[:, 2] - z_min) / (z_max - z_min) - 0.85

    rv = rv - centroid
    rv[:, 0] = 1.7 * (rv[:, 0] - x_min) / (x_max - x_min) - 0.85
    rv[:, 1] = 1.7 * (rv[:, 1] - y_min) / (y_max - y_min) - 0.85
    rv[:, 2] = 1.7 * (rv[:, 2] - z_min) / (z_max - z_min) - 0.85

    return pc, epi_myo, endo_myo, rv
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
def crop_mask_region_and_generate_bbox(sitkImage, sitkMask, **kwargs):
    """
    :param sitkMask:
    :param kwargs: spare_boundary_mm = (1,1,1)
    :return:
    """
    np_imgs = sitk.GetArrayFromImage(sitkMask)
    np_imgs_xy = np.sum(np_imgs, axis=0)
    x_line = np.sum(np_imgs_xy, axis=0)
    y_line = np.sum(np_imgs_xy, axis=1)

    np_imgs_yz = np.sum(np_imgs, axis=2)
    z_line = np.sum(np_imgs_yz, axis=1)

    # 2D image parameters
    slices, height, width = np_imgs.shape

    x_1, x_2 = find_corner_index(x_line, 0.9)
    y_1, y_2 = find_corner_index(y_line, 0.9)
    z_1, z_2 = find_corner_index(z_line, 0.9)

    if z_1 > 9:
        z_1 = z_1 -10
    else:
        z_1 = 0

    if x_1 > 9:
        x_1 = x_1 -10
    else:
        x_1 = 0

    if y_1 > 9:
        y_1 = y_1 -10
    else:
        y_1 = 0

    if slices-1-z_2-10 > -1:
        z_2 = slices-1-z_2-10
    else:
        z_2 = 0

    lowerBoundary = (x_1, y_1, z_1)
    upperBoundary = (width-1-x_2-10, height-1-y_2-10, z_2)

    cropped_images = sitk.Crop(sitkImage, lowerBoundary, upperBoundary)

    cropped_masks = sitk.Crop(sitkMask, lowerBoundary, upperBoundary)

    return cropped_images, cropped_masks


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.random_integers(0, N-1, 1)

    if N < npoint:
        for i in range(N):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            distance = np.minimum(distance, dist)
            farthest = np.argmax(distance)
        for i in range(N, npoint):
            centroids[i] = centroids[i-N]
    else:
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            distance = np.minimum(distance, dist)
            farthest = np.argmax(distance)
    return centroids


src_root1 = '/ailab/data/Cardiac super-resolution label maps/smoothed'
dst_root = '/ailab/data/Cardiac super-resolution label maps/pc1/'

###  number = '1004' CT LAX has a problem


for subroot, dirs, files in os.walk(src_root1):
    for file in files:
        if file.endswith('nii.gz') and 'MYO' in file:
            patient_num = subroot
            # print(file)

            patient_num_vec = patient_num.split(os.path.sep)
            p1 = patient_num_vec[-1]
            print('working on %s', p1)

            mask_file = os.path.join(subroot, file)
            mask_image = sitk.ReadImage(mask_file)
            size = mask_image.GetSize()  # x, y, z
            myo = sitk.GetArrayFromImage(mask_image)

            mask_file = os.path.join(subroot, file.replace('MYO', 'RV'))
            mask_image = sitk.ReadImage(mask_file)
            size = mask_image.GetSize()  # x, y, z
            rv = sitk.GetArrayFromImage(mask_image)


            np_imgs_yz = np.sum(myo, axis=2)
            z_line = np.sum(np_imgs_yz, axis=1)
            z_1, z_2 = find_corner_index(z_line, 0.9)

            m = 10
            th = (z_2 - z_1) / (m + 1)
            m_pc1 = np.zeros([m + 4, 400, 3])

            # collect the SAX slices
            for s in range(m):
                z = z_1 + (s + 1) * th
                z = int(z)
                ## for myo and rv
                contours_rv = measure.find_contours(rv[z, ::], level=0.8)
                contours = measure.find_contours(myo[z, ::], level=0.8)
                num = np.zeros(len(contours))
                for k in range(len(contours)):
                    num[k] = contours[k].shape[0]
                c_ind = num.argsort()

                # fig = plt.figure(figsize=(3, 3), dpi=300)
                # for n, c in enumerate(contours):
                #     plt.plot(c[:, 1], c[:, 0])
                # plt.imshow(myo[z, ::], cmap='gray')
                # plt.axis('off')
                # plt.margins(0, 0)
                # plt.show()
                # plt.close()

                if len(contours_rv) < 1:
                    lv_num = 400
                else:
                    lv_num = 200

                if len(contours) > 1:
                    c_epi = np.asarray(contours[c_ind[-1]])
                    c_endo = np.asarray(contours[c_ind[-2]])
                    if c_endo.shape[0] > 20:
                        n_endo = c_endo.shape[0]
                        n_epi = c_epi.shape[0]
                        n_endo = n_endo/ (n_endo + n_epi) *lv_num
                        n_endo = int(n_endo)
                        c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num-n_endo)
                        c_endo_1 = farthest_point_sample(c_endo, npoint=n_endo)
                        c_epi_1 = c_epi[c_epi_1]
                        c_endo_1 = c_endo[c_endo_1]
                        m_pc1[s, :(lv_num-n_endo), :2] = c_epi_1
                        m_pc1[s, :, 2] = z
                        m_pc1[s, (lv_num-n_endo):lv_num, :2] = c_endo_1
                    else:
                        c_epi = np.asarray(contours[c_ind[-1]])
                        c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                        c_epi_1 = c_epi[c_epi_1]
                        m_pc1[s, :lv_num, :2] = c_epi_1
                        m_pc1[s, :, 2] = z

                else:
                    c_epi = np.asarray(contours[0])
                    # print(1)
                    # print(c_epi.shape)
                    c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                    c_epi_1 = c_epi[c_epi_1]
                    m_pc1[s, :lv_num, :2] = c_epi_1
                    m_pc1[s, :, 2] = z

                ## for rv
                if lv_num < 201:
                    contours = measure.find_contours(rv[z, ::], level=0.8)
                    num = np.zeros(len(contours))
                    for k in range(len(contours)):
                        num[k] = contours[k].shape[0]
                    c_ind = num.argsort()

                    # fig = plt.figure(figsize=(3, 3), dpi=300)
                    # for n, c in enumerate(contours):
                    #     plt.plot(c[:, 1], c[:, 0])
                    # plt.imshow(myo[z, ::], cmap='gray')
                    # plt.axis('off')
                    # plt.margins(0, 0)
                    # plt.show()
                    # plt.close()

                    if len(contours) > 1:
                        c_epi = np.asarray(contours[c_ind[-1]])
                        c_endo = np.asarray(contours[c_ind[-2]])
                        if c_endo.shape[0] > 20:
                            n_endo = c_endo.shape[0]
                            n_epi = c_epi.shape[0]
                            n_endo = n_endo / (n_endo + n_epi) * lv_num
                            n_endo = int(n_endo)
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num - n_endo)
                            c_endo_1 = farthest_point_sample(c_endo, npoint=n_endo)
                            c_epi_1 = c_epi[c_epi_1]
                            c_endo_1 = c_endo[c_endo_1]
                            m_pc1[s, lv_num:(lv_num + lv_num - n_endo), :2] = c_epi_1
                            m_pc1[s, :, 2] = z
                            m_pc1[s, (lv_num + lv_num - n_endo):, :2] = c_endo_1
                        else:
                            c_epi = np.asarray(contours[c_ind[-1]])
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                            c_epi_1 = c_epi[c_epi_1]
                            m_pc1[s, lv_num:, :2] = c_epi_1
                            m_pc1[s, :, 2] = z

                    else:
                        c_epi = np.asarray(contours[0])
                        # print(1)
                        # print(c_epi.shape)
                        c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                        c_epi_1 = c_epi[c_epi_1]
                        m_pc1[s, lv_num:, :2] = c_epi_1
                        m_pc1[s, :, 2] = z

            # m_pcx = np.concatenate(m_pc1, axis=0)
            # fig = plt.figure(figsize=(3, 3))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(m_pcx[:4000, 0], m_pcx[:4000, 1], m_pcx[:4000, 2], s=2, c='blue')
            # ax.axis()
            # plt.show()

            np_imgs_yz = np.sum(rv, axis=2)
            z_line = np.sum(np_imgs_yz, axis=1)
            z_1, z_2 = find_corner_index(z_line, 0.9)
            z_c = int(z_1 + (z_2 - z_1)/3)

            lv_c = myo[z_c, ::]
            rv_c = rv[z_c, ::]
            lv_mask_value = 2
            rv_mask_value = 3
            lv_idx = np.where(lv_c == lv_mask_value)  # z, y, x
            rv_idx = np.where(rv_c == rv_mask_value)  # z, y, x
            lv_p = [np.mean(lv_idx[0]), np.mean(lv_idx[1])]
            rv_p = [np.mean(rv_idx[0]), np.mean(rv_idx[1])]

            d_vec = np.asarray(lv_p) - np.asarray(rv_p)
            if d_vec[0] < 0:
                d_vec[0] = d_vec[0] - 0.0001
            else:
                d_vec[0] = d_vec[0] + 0.0001

            if d_vec[1] < 0:
                d_vec[1] = d_vec[1] - 0.0001
            else:
                d_vec[1] = d_vec[1] + 0.0001

            ang_0 = np.arctan(d_vec[1] / d_vec[0])

            if d_vec[0] < 0 and d_vec[1] > 0:
                z_ang = - np.pi - ang_0
            elif d_vec[0] > 0 and d_vec[1] > 0:
                z_ang = - ang_0
            elif d_vec[0] > 0 and d_vec[1] < 0:
                z_ang = - ang_0
            elif d_vec[0] < 0 and d_vec[1] < 0:
                z_ang = np.pi - ang_0



            shape = myo.shape
            lax_pc = np.zeros([3, 400, 3])

            for s in range(m, m + 3):
                ang = z_ang - np.pi / 2 - (s - m) * np.pi / 3 * 2 - np.pi * random.uniform(-0.02, 0.02)
                k_tan_ang = np.tan(ang)
                # print(k)
                if abs(k_tan_ang) < 1:
                    x = np.floor(np.linspace(0, shape[-1]-1, shape[-1]))
                    y = np.floor(k_tan_ang*(x-lv_p[1]) + lv_p[0])
                    xf = np.linspace(0, shape[-1] - 1, shape[-1])
                    yf = k_tan_ang * (x - lv_p[1]) + lv_p[0]

                    x = x[np.where(y > -1)]
                    xf = xf[np.where(y > -1)]
                    yf = yf[np.where(y > -1)]
                    y = y[np.where(y > -1)]

                    x = x[np.where(y < shape[-2])]
                    xf = xf[np.where(y < shape[-2])]
                    yf = yf[np.where(y < shape[-2])]
                    y = y[np.where(y < shape[-2])]

                    x = x.astype(int)
                    y = y.astype(int)


                else:
                    y = np.floor(np.linspace(0, shape[-2] - 1, shape[-2]))
                    x = np.floor(1/k_tan_ang * (y - lv_p[0]) + lv_p[1])
                    yf = np.linspace(0, shape[-2] - 1, shape[-2])
                    xf = 1 / k_tan_ang * (y - lv_p[0]) + lv_p[1]

                    y = y[np.where(x > -1)]
                    yf = yf[np.where(x > -1)]
                    xf = xf[np.where(x > -1)]
                    x = x[np.where(x > -1)]

                    y = y[np.where(x < shape[-2])]
                    yf = yf[np.where(x < shape[-2])]
                    xf = xf[np.where(x < shape[-2])]
                    x = x[np.where(x < shape[-2])]


                    x = x.astype(int)
                    y = y.astype(int)



                if s != m:
                    f_ch = myo[:, y, x]

                    contours = measure.find_contours(f_ch, level=0.8)

                    # fig = plt.figure(figsize=(3, 3), dpi=300)
                    # for n, c in enumerate(contours):
                    #     plt.plot(c[:, 1], c[:, 0])
                    # plt.imshow(f_ch, cmap='gray')
                    # plt.axis('off')
                    # plt.margins(0, 0)
                    # plt.show()
                    # plt.close()

                    num = np.zeros(len(contours))
                    for k in range(len(contours)):
                        num[k] = contours[k].shape[0]
                    c_ind = num.argsort()

                    lv_num = 400

                    if len(contours) > 1:
                        c_epi = np.asarray(contours[c_ind[-1]])
                        c_endo = np.asarray(contours[c_ind[-2]])
                        if c_endo.shape[0] > 20:
                            n_endo = c_endo.shape[0]
                            n_epi = c_epi.shape[0]
                            n_endo = n_endo / (n_endo + n_epi) * lv_num
                            n_endo = int(n_endo)
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num - n_endo)
                            c_endo_1 = farthest_point_sample(c_endo, npoint=n_endo)
                            c_epi_1 = c_epi[c_epi_1]
                            c_endo_1 = c_endo[c_endo_1]

                            xy_index = np.floor(c_epi_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[s, :(lv_num - n_endo), 1] = xf[xy_index]
                            m_pc1[s, :(lv_num - n_endo), 0] = yf[xy_index]
                            m_pc1[s, :(lv_num - n_endo), 2] = c_epi_1[:, 0]

                            xy_index = np.floor(c_endo_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[s, (lv_num - n_endo):, 1] = xf[xy_index]
                            m_pc1[s, (lv_num - n_endo):, 0] = yf[xy_index]
                            m_pc1[s, (lv_num - n_endo):, 2] = c_endo_1[:, 0]

                        else:
                            c_epi = np.asarray(contours[c_ind[-1]])
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                            c_epi_1 = c_epi[c_epi_1]
                            xy_index = np.floor(c_epi_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[s, :, 1] = xf[xy_index]
                            m_pc1[s, :, 0] = yf[xy_index]
                            m_pc1[s, :, 2] = c_epi_1[:, 0]

                    else:
                        c_epi = np.asarray(contours[0])
                        # print(1)
                        # print(c_epi.shape)
                        c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                        c_epi_1 = c_epi[c_epi_1]

                        xy_index = np.floor(c_epi_1[:, 1])
                        xy_index = xy_index.astype(int)
                        m_pc1[s, :, 1] = xf[xy_index]
                        m_pc1[s, :, 0] = yf[xy_index]
                        m_pc1[s, :, 2] = c_epi_1[:, 0]

                else:
                    f_ch = myo[:, y, x]
                    contours = measure.find_contours(f_ch, level=0.8)

                    # fig = plt.figure(figsize=(3, 3), dpi=300)
                    # for n, c in enumerate(contours):
                    #     plt.plot(c[:, 1], c[:, 0])
                    # plt.imshow(f_ch, cmap='gray')
                    # plt.axis('off')
                    # plt.margins(0, 0)
                    # plt.show()
                    # plt.close()

                    num = np.zeros(len(contours))
                    for k in range(len(contours)):
                        num[k] = contours[k].shape[0]
                    c_ind = num.argsort()

                    lv_num = 400

                    if len(contours) > 1:
                        c_epi = np.asarray(contours[c_ind[-1]])
                        c_endo = np.asarray(contours[c_ind[-2]])
                        if c_endo.shape[0] > 20:
                            n_endo = c_endo.shape[0]
                            n_epi = c_epi.shape[0]
                            n_endo = n_endo / (n_endo + n_epi) * lv_num
                            n_endo = int(n_endo)
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num - n_endo)
                            c_endo_1 = farthest_point_sample(c_endo, npoint=n_endo)
                            c_epi_1 = c_epi[c_epi_1]
                            c_endo_1 = c_endo[c_endo_1]

                            xy_index = np.floor(c_epi_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[s, :(lv_num - n_endo), 1] = xf[xy_index]
                            m_pc1[s, :(lv_num - n_endo), 0] = yf[xy_index]
                            m_pc1[s, :(lv_num - n_endo), 2] = c_epi_1[:, 0]

                            xy_index = np.floor(c_endo_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[s, (lv_num - n_endo): lv_num, 1] = xf[xy_index]
                            m_pc1[s, (lv_num - n_endo): lv_num, 0] = yf[xy_index]
                            m_pc1[s, (lv_num - n_endo): lv_num, 2] = c_endo_1[:, 0]

                        else:
                            c_epi = np.asarray(contours[c_ind[-1]])
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                            c_epi_1 = c_epi[c_epi_1]
                            xy_index = np.floor(c_epi_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[s, :lv_num, 1] = xf[xy_index]
                            m_pc1[s, :lv_num, 0] = yf[xy_index]
                            m_pc1[s, :lv_num, 2] = c_epi_1[:, 0]

                    else:
                        c_epi = np.asarray(contours[0])
                        # print(1)
                        # print(c_epi.shape)
                        c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                        c_epi_1 = c_epi[c_epi_1]

                        xy_index = np.floor(c_epi_1[:, 1])
                        xy_index = xy_index.astype(int)
                        m_pc1[s, :lv_num, 1] = xf[xy_index]
                        m_pc1[s, :lv_num, 0] = yf[xy_index]
                        m_pc1[s, :lv_num, 2] = c_epi_1[:, 0]

                    f_ch = rv[:, y, x]
                    contours = measure.find_contours(f_ch, level=0.8)

                    # fig = plt.figure(figsize=(3, 3), dpi=300)
                    # for n, c in enumerate(contours):
                    #     plt.plot(c[:, 1], c[:, 0])
                    # plt.imshow(f_ch, cmap='gray')
                    # plt.axis('off')
                    # plt.margins(0, 0)
                    # plt.show()
                    # plt.close()

                    num = np.zeros(len(contours))
                    for k in range(len(contours)):
                        num[k] = contours[k].shape[0]
                    c_ind = num.argsort()

                    if len(contours) > 1:
                        c_epi = np.asarray(contours[c_ind[-1]])
                        c_endo = np.asarray(contours[c_ind[-2]])
                        if c_endo.shape[0] > 20:
                            n_endo = c_endo.shape[0]
                            n_epi = c_epi.shape[0]
                            n_endo = n_endo / (n_endo + n_epi) * lv_num
                            n_endo = int(n_endo)
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num - n_endo)
                            c_endo_1 = farthest_point_sample(c_endo, npoint=n_endo)
                            c_epi_1 = c_epi[c_epi_1]
                            c_endo_1 = c_endo[c_endo_1]

                            xy_index = np.floor(c_epi_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[-1, :(lv_num - n_endo), 1] = xf[xy_index]
                            m_pc1[-1, :(lv_num - n_endo), 0] = yf[xy_index]
                            m_pc1[-1, :(lv_num - n_endo), 2] = c_epi_1[:, 0]

                            xy_index = np.floor(c_endo_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[-1, (lv_num - n_endo): lv_num, 1] = xf[xy_index]
                            m_pc1[-1, (lv_num - n_endo): lv_num, 0] = yf[xy_index]
                            m_pc1[-1, (lv_num - n_endo): lv_num, 2] = c_endo_1[:, 0]

                        else:
                            c_epi = np.asarray(contours[c_ind[-1]])
                            c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                            c_epi_1 = c_epi[c_epi_1]
                            xy_index = np.floor(c_epi_1[:, 1])
                            xy_index = xy_index.astype(int)
                            m_pc1[-1, :lv_num, 1] = xf[xy_index]
                            m_pc1[-1, :lv_num, 0] = yf[xy_index]
                            m_pc1[-1, :lv_num, 2] = c_epi_1[:, 0]

                    else:
                        c_epi = np.asarray(contours[0])
                        # print(1)
                        # print(c_epi.shape)
                        c_epi_1 = farthest_point_sample(c_epi, npoint=lv_num)
                        c_epi_1 = c_epi[c_epi_1]

                        xy_index = np.floor(c_epi_1[:, 1])
                        xy_index = xy_index.astype(int)
                        m_pc1[-1, :lv_num, 1] = xf[xy_index]
                        m_pc1[-1, :lv_num, 0] = yf[xy_index]
                        m_pc1[-1, :lv_num, 2] = c_epi_1[:, 0]


            m_pc = orientation_correction(m_pc1)
            m_pc = np.concatenate(m_pc, axis=0)

            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(m_pc[:, 0], m_pc[:, 1], m_pc[:, 2], s=2, c='green')
            # ax.scatter(m_pc[4000:, 0], m_pc[4000:, 1], m_pc[4000:, 2], s=2, c='green')
            # ax.scatter(epi_myo[:, 0], epi_myo[:, 1], epi_myo[:, 2], s=2, c='red')
            # ax.scatter(endo_myo[:, 0], endo_myo[:, 1], endo_myo[:, 2], s=2, c='blue')
            # ax.scatter(rv[:, 0], rv[:, 1], rv[:, 2], s=2, c='yellow')
            ax.axis()
            plt.show()



            save_dir = os.path.join(dst_root, p1)
            if not os.path.exists(save_dir): os.mkdir(save_dir)
            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'pc_0.npz'))
            np.savez_compressed(pc_np_file, m_pc)
            # print('m_pc.shape')
            # print(m_pc.shape)



            epi_myo_file = os.path.join(subroot, file.replace('MYO.nii.gz', 'MYO_epi2.npz'))
            endo_myo_file = os.path.join(subroot, file.replace('MYO.nii.gz', 'MYO_endo2.npz')) #'HR_ES_MYO_endo2.npz')
            rv_file = os.path.join(subroot, file.replace('MYO.nii.gz', 'RV_pc.mat')) #'HR_ES_RV_pc.mat')

            npz = np.load(epi_myo_file)
            np_arrs = []
            for np_file in npz.keys(): np_arrs.append(npz[np_file])
            epi_myo_data = np_arrs[0]

            npz = np.load(endo_myo_file)
            np_arrs = []
            for np_file in npz.keys(): np_arrs.append(npz[np_file])
            endo_myo_data = np_arrs[0]
            rv_data = scio.loadmat(rv_file)

            epi_myo = orientation_correction1(epi_myo_data)-1
            endo_myo = orientation_correction1(endo_myo_data)-1
            rv_pc = orientation_correction1(rv_data['pc'])-1

            rv = rv_pc[farthest_point_sample(rv_pc, 3000)]
            epi_myo = epi_myo[farthest_point_sample(epi_myo, 3000)]
            endo_myo = endo_myo[farthest_point_sample(endo_myo, 3000)]

            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'epi_myo_0.npz'))
            np.savez_compressed(pc_np_file, epi_myo)

            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'endo_myo_0.npz'))
            np.savez_compressed(pc_np_file, endo_myo)

            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'rv_0.npz'))
            np.savez_compressed(pc_np_file, rv)

            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(m_pc[:, 0], m_pc[:, 1], m_pc[:, 2], s=2, c='green')
            # ax.scatter(m_pc[4000:, 0], m_pc[4000:, 1], m_pc[4000:, 2], s=2, c='green')
            ax.scatter(epi_myo[:, 0], epi_myo[:, 1], epi_myo[:, 2], s=2, c='red')
            ax.scatter(endo_myo[:, 0], endo_myo[:, 1], endo_myo[:, 2], s=2, c='blue')
            ax.scatter(rv[:, 0], rv[:, 1], rv[:, 2], s=2, c='yellow')
            ax.axis()
            plt.show()

            print(12 / 0)

            m_pc, epi_myo, endo_myo, rv = pc_normalize(m_pc[:, :3], epi_myo, endo_myo, rv)

            r = R.from_euler('xyz', (0, 0, z_ang), degrees=False)
            m_pc = r.apply(m_pc)  # Rotated points
            epi_myo = r.apply(epi_myo)  # Rotated points
            endo_myo = r.apply(endo_myo)  # Rotated points
            rv = r.apply(rv)  # Rotated points

            np.savetxt(os.path.join(save_dir, file.replace('MYO.nii.gz', 'z_ang.txt')), [z_ang], fmt='%.6f')

            m_pc, epi_myo, endo_myo, rv = pc_normalize(m_pc, epi_myo, endo_myo, rv)

            # fig = plt.figure(figsize=(3, 3))
            # ax = fig.add_subplot(111, projection='3d')
            # # ax.scatter(m_pc[5200:, 0], m_pc[5200:, 1], m_pc[5200:, 2], s=2, c='green')
            # ax.scatter(m_pc[4000:, 0], m_pc[4000:, 1], m_pc[4000:, 2], s=2, c='green')
            # ax.scatter(epi_myo[:, 0], epi_myo[:, 1], epi_myo[:, 2], s=2, c='red')
            # ax.scatter(endo_myo[:, 0], endo_myo[:, 1], endo_myo[:, 2], s=2, c='blue')
            # ax.scatter(rv[:, 0], rv[:, 1], rv[:, 2], s=2, c='yellow')
            # ax.axis()
            # plt.show()


            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'pc.npz'))
            np.savez_compressed(pc_np_file, m_pc)

            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'epi_myo.npz'))
            np.savez_compressed(pc_np_file, epi_myo)

            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'endo_myo.npz'))
            np.savez_compressed(pc_np_file, endo_myo)

            pc_np_file = os.path.join(save_dir, file.replace('MYO.nii.gz', 'rv.npz'))
            np.savez_compressed(pc_np_file, rv)

            # print(12/0)



