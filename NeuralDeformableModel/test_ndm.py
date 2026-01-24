"""
Author: Meng Ye
Date: Nov 2022
"""
import os
import torch
import math
from torch.autograd import Variable
import logging
import scipy.io as scio
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import random
from torch import optim
import numpy as np
from pytorch3d.loss import chamfer_distance
from model.model import NeuralDeformableModel

from config import get_config

quaternion_loss = torch.nn.MSELoss()

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
    # farthest = np.random.random_integers(0, N-1, 1)
    farthest = np.random.randint(0, N - 1, 1)

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

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# quaternion batch*4
def compute_rotation_matrix_from_quaternion(quaternion):
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion)

    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix, quat

def smooothing_loss(y_pred):
    # y_pred is the deformation_matrix of shape (batch_size, seq_length, channels=2, height, width)
    dy = torch.abs(y_pred[:, 1:, :] - y_pred[:, :-1, :])
   

    dy = torch.mul(dy, dy)
    d = torch.sqrt(torch.mean(dy))
    return d

def displacement_smooothing_loss(primitive, deformed_primitive, Nr=100):
    # primitive, deformed_primitive: B, N, 3
    # y_pred is the deformation_matrix of shape (B, H, W, channels=3)
    y_pred = deformed_primitive - primitive
    s = deformed_primitive.shape
    y_pred = y_pred.view([s[0], -1, Nr, s[-1]])
    dx = torch.abs(y_pred[:, 1:, ::] - y_pred[:, :-1, ::])
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    # dxy = torch.mul(dx, dx) + torch.mul(dy, dy)
    d = torch.sqrt(torch.mean(torch.mul(dx, dx))) + torch.sqrt(torch.mean(torch.mul(dy, dy)))
    return d

def add_lids_to_a3(m_pc1, n_theta=50, n=5):
    m_pc1 = m_pc1.view(-1, n_theta, 1)
    s = m_pc1.shape
    m_pc11 = torch.zeros([s[0], s[1]+ n, s[2]]).cuda()
    m_pc11[:, :s[1], :] = m_pc1
    for i in range(n):
        m_pc11[:, s[1] + i, :] = m_pc11[:, s[1] + i - 1, :]

    return m_pc11

def add_lids_to_a1(m_pc1, n_theta=50, n=5, a=0.95):
    m_pc1 = m_pc1.view(-1, n_theta, 1)
    s = m_pc1.shape
    m_pc11 = torch.zeros([s[0], s[1]+ n, s[2]]).cuda()
    m_pc11[:, :s[1], :] = m_pc1
    for i in range(n):
        m_pc11[:, s[1] + i, :] = a * m_pc11[:, s[1] + i - 1, :]

    return m_pc11

# device configuration
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main():
    '''HYPER PARAMETER'''
    logger = logging.getLogger(__name__)

    '''MODEL LOADING'''
    print('Begin MODEL LOADING')

    model_path = '/ailab/models/neural_deformable_model1/'
    if not os.path.exists(model_path): os.makedirs(model_path)

    # **** BUILD NMF MODEL ******
    experiment, opt = get_config()
    GeoMorph = NeuralDeformableModel(zdim=512, time=opt.toi, tol=opt.tolerance)
    print('Finish MODEL LOADING')
    GeoMorph = GeoMorph.eval()  
    GeoMorph.cuda()
    GeoMorph = GeoMorph.float()

    pretrained_model = '720_0.0037_model.pth'

    GeoMorph.load_state_dict(torch.load(os.path.join(model_path, pretrained_model)))
    epoch = int(pretrained_model.split('_')[0])
    print('test epoch:', epoch)

    # q_prior = torch.tensor([-0.9421, 0.0605, -0.0373, 0.3277])
    src_root = '/ailab/data/Cardiac super-resolution label maps/pc1/'


    samples_in_epoch = [1327]

    it_num = 50

    sw = 0.0

    q_prior = torch.tensor([-0.9956, 0.0515, 0.0335, -0.0711])

    Npoints_epi = 5500
    Npoints_endo = 5000
    Nr = 100
    Npoints_rv = 5000

    for i in samples_in_epoch:
        logger.info('Begin to generate the training sample')
        j = i
        if np.random.rand() > 0.5:
            print('ED')
            epi_myo_file = os.path.join(src_root, str(j), 'HR_ED_epi_myo.npz')
            endo_myo_file = os.path.join(src_root, str(j), 'HR_ED_endo_myo.npz')
            rv_file = os.path.join(src_root, str(j), 'HR_ED_rv.npz')
            pc_file = os.path.join(src_root, str(j), 'HR_ED_pc.npz')
        else:
            print('ES')
            epi_myo_file = os.path.join(src_root, str(j), 'HR_ES_epi_myo.npz')
            endo_myo_file = os.path.join(src_root, str(j), 'HR_ES_endo_myo.npz')
            rv_file = os.path.join(src_root, str(j), 'HR_ES_rv.npz')
            pc_file = os.path.join(src_root, str(j), 'HR_ES_pc.npz')

        npz = np.load(epi_myo_file)
        np_arrs = []
        for np_file in npz.keys(): np_arrs.append(npz[np_file])
        epi_myo_gt0 = np_arrs[0]

        npz = np.load(endo_myo_file)
        np_arrs = []
        for np_file in npz.keys(): np_arrs.append(npz[np_file])
        endo_myo_gt0 = np_arrs[0]

        npz = np.load(rv_file)
        np_arrs = []
        for np_file in npz.keys(): np_arrs.append(npz[np_file])
        rv_gt0 = np_arrs[0]

        npz = np.load(pc_file)
        np_arrs = []
        for np_file in npz.keys(): np_arrs.append(npz[np_file])
        m_pc0 = np_arrs[0]

        m_pc = np.expand_dims(m_pc0, axis=0)
        epi_myo_gt = np.expand_dims(epi_myo_gt0, axis=0)
        endo_myo_gt = np.expand_dims(endo_myo_gt0, axis=0)
        rv_gt = np.expand_dims(rv_gt0, axis=0)

        logger.info('Finshed generating the training sample')

        points = torch.Tensor(m_pc)
        epi_myo_gt = torch.Tensor(epi_myo_gt)
        endo_myo_gt = torch.Tensor(endo_myo_gt)
        rv_gt = torch.Tensor(rv_gt)
        points = to_var(points)
        epi_myo_gt = to_var(epi_myo_gt)
        endo_myo_gt = to_var(endo_myo_gt)
        rv_gt = to_var(rv_gt)
        points, epi_myo_gt, endo_myo_gt, rv_gt = points.float().cuda(), epi_myo_gt.float().cuda(), endo_myo_gt.float().cuda(), rv_gt.float().cuda()

        code1, code2, code3, trans, quaternion, scale, a1, a2, a3, e1, e2, trans2, quaternion2, scale2, a12, a22, a32, e12, e22, \
        trans3, quaternion3, scale3, a13, a23, a33, e13, e23, a14, sph = GeoMorph.forward(points)

        a1 = add_lids_to_a1(a1)
        a2 = add_lids_to_a1(a2)
        a3 = add_lids_to_a3(a3)
        e1 = add_lids_to_a3(e1)
        e2 = add_lids_to_a3(e2)

        epi_pc = sph[:, :Npoints_epi, :]
        endo_pc = sph[:, Npoints_epi:Npoints_epi + Npoints_endo, :]
        rv = sph[:, Npoints_epi + Npoints_endo:, :]

        if epoch < it_num + 1:
            # print('translation')
            sm_loss1 = quaternion_loss(trans, trans2)
            sm_loss2 = 0

            epi_pc = epi_pc + trans
            endo_pc = endo_pc + trans2
            rv = rv + trans3

            loss111, _ = chamfer_distance(epi_pc, epi_myo_gt)
            loss112, _ = chamfer_distance(endo_pc, endo_myo_gt)
            loss11 = loss111 + loss112
            loss12, _ = chamfer_distance(rv, rv_gt)
            loss = loss11 + loss12 + sw * (sm_loss1 + sm_loss2)

        elif epoch < 2 * it_num + 1:
            # print('scaling')
            sm_loss1 = quaternion_loss(scale, scale2) + quaternion_loss(trans, trans2)
            sm_loss2 = 0

            epi_pc = scale * epi_pc
            epi_pc = epi_pc + trans

            endo_pc = scale2 * endo_pc
            endo_pc = endo_pc + trans2

            rv = scale3 * rv
            rv = rv + trans3

            loss111, _ = chamfer_distance(epi_pc, epi_myo_gt)
            loss112, _ = chamfer_distance(endo_pc, endo_myo_gt)
            loss11 = loss111 + loss112
            loss12, _ = chamfer_distance(rv, rv_gt)
            loss = loss11 + loss12 + sw * (sm_loss1 + sm_loss2)

        elif epoch < 3 * it_num + 1:
            # print('rotation')
            rotation_m, quaternion = compute_rotation_matrix_from_quaternion(quaternion)  # Bx3x3
            rotation_m2, quaternion2 = compute_rotation_matrix_from_quaternion(quaternion2)  # Bx3x3
            rotation_m3, quaternion3 = compute_rotation_matrix_from_quaternion(quaternion3)  # Bx3x3

            sm_loss1 = quaternion_loss(quaternion, quaternion2) + quaternion_loss(scale, scale2) + quaternion_loss(
                trans, trans2)
            sm_loss2 = quaternion_loss(quaternion, quaternion3)

            # print('quaternion')
            # print(quaternion)
            # print('quaternion2')
            # print(quaternion2)
            # qloss = quaternion_loss(quaternion, quaternion2)
            qloss_sw = 2 - 0.2 * (epoch - 2 * it_num - 1)

            if qloss_sw < 0.0001:
                qloss = 0
                # print('qloss_sw')
                # print(qloss_sw)
            else:
                q_prior = q_prior.cuda()
                s = quaternion.shape
                q_prior = q_prior.expand(s[0], s[1])
                qloss = quaternion_loss(q_prior, quaternion) + \
                        quaternion_loss(q_prior, quaternion2) + \
                        quaternion_loss(q_prior, quaternion3)
                # print('qloss_sw')
                # print(qloss_sw)
                # print('qloss')
                # print(qloss)
                qloss = qloss_sw * qloss

            epi_pc = scale * epi_pc
            epi_pc = torch.bmm(epi_pc, torch.transpose(rotation_m, 1, 2))
            epi_pc = epi_pc + trans

            endo_pc = scale2 * endo_pc
            endo_pc = torch.bmm(endo_pc, torch.transpose(rotation_m2, 1, 2))
            endo_pc = endo_pc + trans2

            rv = scale3 * rv
            rv = torch.bmm(rv, torch.transpose(rotation_m3, 1, 2))
            rv = rv + trans3

            loss111, _ = chamfer_distance(epi_pc, epi_myo_gt)
            loss112, _ = chamfer_distance(endo_pc, endo_myo_gt)
            loss11 = loss111 + loss112
            loss12, _ = chamfer_distance(rv, rv_gt)

            loss = loss11 + loss12 + sw * (sm_loss1 + sm_loss2) + qloss

        elif epoch < 5 * it_num + 1:
            # print('a3')
            rotation_m, quaternion = compute_rotation_matrix_from_quaternion(quaternion)  # Bx3x3
            rotation_m2, quaternion2 = compute_rotation_matrix_from_quaternion(quaternion2)  # Bx3x3
            rotation_m3, quaternion3 = compute_rotation_matrix_from_quaternion(quaternion3)  # Bx3x3

            sm_loss1 = smooothing_loss(a3) + quaternion_loss(quaternion, quaternion2) + quaternion_loss(scale,
                                                                                                        scale2) + quaternion_loss(
                trans, trans2)
            sm_loss2 = smooothing_loss(a32) + quaternion_loss(quaternion, quaternion3)
            sm_loss3 = smooothing_loss(a33)

            a3 = a3.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            a1 = torch.ones(a3.shape).cuda()
            a2 = torch.ones(a3.shape).cuda()
            epi_pc = torch.cat([a1, a2, a3], dim=-1) * epi_pc
            epi_pc = scale * epi_pc
            epi_pc = torch.bmm(epi_pc, torch.transpose(rotation_m, 1, 2))
            epi_pc = epi_pc + trans

            a32 = a32.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            a12 = torch.ones(a32.shape).cuda()
            a22 = torch.ones(a32.shape).cuda()
            endo_pc = torch.cat([a12, a22, a32], dim=-1) * endo_pc
            endo_pc = scale2 * endo_pc
            endo_pc = torch.bmm(endo_pc, torch.transpose(rotation_m2, 1, 2))
            endo_pc = endo_pc + trans2

            a33 = a33.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            a13 = torch.ones(a33.shape).cuda()
            a23 = torch.ones(a33.shape).cuda()
            rv = torch.cat([a13, a23, a33], dim=-1) * rv
            rv = scale3 * rv
            rv = torch.bmm(rv, torch.transpose(rotation_m3, 1, 2))
            rv = rv + trans3

            loss111, _ = chamfer_distance(epi_pc, epi_myo_gt)
            loss112, _ = chamfer_distance(endo_pc, endo_myo_gt)
            loss11 = loss111 + loss112
            loss12, _ = chamfer_distance(rv, rv_gt)
            loss = loss11 + loss12 + sw * (sm_loss1 + sm_loss2 + sm_loss3)

        elif epoch < 9 * it_num + 1:
            # print('e1, e2')
            rotation_m, quaternion = compute_rotation_matrix_from_quaternion(quaternion)  # Bx3x3
            rotation_m2, quaternion2 = compute_rotation_matrix_from_quaternion(quaternion2)  # Bx3x3
            rotation_m3, quaternion3 = compute_rotation_matrix_from_quaternion(quaternion3)  # Bx3x3

            sm_loss1 = smooothing_loss(e1) + smooothing_loss(e2) + smooothing_loss(a3) + quaternion_loss(quaternion,
                                                                                                         quaternion2) + quaternion_loss(
                scale, scale2) + quaternion_loss(trans, trans2)
            sm_loss2 = smooothing_loss(e12) + smooothing_loss(e22) + smooothing_loss(a32) + quaternion_loss(quaternion,
                                                                                                            quaternion3)
            sm_loss3 = smooothing_loss(e13) + smooothing_loss(e23) + smooothing_loss(a33)

            e3 = torch.zeros(e1.shape).cuda()
            e1 = e1.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e2 = e2.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            a3 = a3.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            a1 = torch.ones(a3.shape).cuda()
            a2 = torch.ones(a3.shape).cuda()
            epi_pc = torch.cat([a1, a2, a3], dim=-1) * epi_pc
            epi_pc = scale * epi_pc
            epi_pc = epi_pc + e
            epi_pc = torch.bmm(epi_pc, torch.transpose(rotation_m, 1, 2))
            epi_pc = epi_pc + trans

            e3 = torch.zeros(e12.shape).cuda()
            e1 = e12.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e2 = e22.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            a32 = a32.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            a12 = torch.ones(a32.shape).cuda()
            a22 = torch.ones(a32.shape).cuda()
            endo_pc = torch.cat([a12, a22, a32], dim=-1) * endo_pc
            endo_pc = scale2 * endo_pc
            endo_pc = endo_pc + e
            endo_pc = torch.bmm(endo_pc, torch.transpose(rotation_m2, 1, 2))
            endo_pc = endo_pc + trans2

            e3 = torch.zeros(e13.shape).cuda()
            e1 = e13.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            e2 = e23.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            a33 = a33.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            a13 = torch.ones(a33.shape).cuda()
            a23 = torch.ones(a33.shape).cuda()
            rv = torch.cat([a13, a23, a33], dim=-1) * rv
            rv = scale3 * rv
            rv = rv + e
            rv = torch.bmm(rv, torch.transpose(rotation_m3, 1, 2))
            rv = rv + trans3

            loss111, _ = chamfer_distance(epi_pc, epi_myo_gt)
            loss112, _ = chamfer_distance(endo_pc, endo_myo_gt)
            loss11 = loss111 + loss112
            loss12, _ = chamfer_distance(rv, rv_gt)
            loss = loss11 + loss12 + sw * (sm_loss1 + sm_loss2 + sm_loss3)


        elif epoch < 13 * it_num + 1:
            # print('a1, a2')
            rotation_m, quaternion = compute_rotation_matrix_from_quaternion(quaternion)  # Bx3x3
            rotation_m2, quaternion2 = compute_rotation_matrix_from_quaternion(quaternion2)  # Bx3x3
            rotation_m3, quaternion3 = compute_rotation_matrix_from_quaternion(quaternion3)  # Bx3x3

            sm_loss1 = smooothing_loss(a1) + smooothing_loss(a2) + smooothing_loss(e1) + smooothing_loss(
                e2) + smooothing_loss(a3) + quaternion_loss(quaternion, quaternion2) + quaternion_loss(scale,
                                                                                                       scale2) + quaternion_loss(
                trans, trans2)
            sm_loss2 = smooothing_loss(a12) + smooothing_loss(a22) + smooothing_loss(e12) + smooothing_loss(
                e22) + smooothing_loss(a32) + quaternion_loss(quaternion, quaternion3)
            sm_loss3 = smooothing_loss(a14) + smooothing_loss(a13) + smooothing_loss(a23) + smooothing_loss(
                e13) + smooothing_loss(e23) + smooothing_loss(a33)

            a1 = a1.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            a2 = a2.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            a3 = a3.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e3 = torch.zeros(e1.shape).cuda()
            e1 = e1.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e2 = e2.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            epi_pc = torch.cat([a1, a2, a3], dim=-1) * epi_pc
            epi_pc = scale * epi_pc
            epi_pc = epi_pc + e
            epi_pc = torch.bmm(epi_pc, torch.transpose(rotation_m, 1, 2))
            epi_pc = epi_pc + trans

            a12 = a12.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            a22 = a22.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            a32 = a32.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e3 = torch.zeros(e12.shape).cuda()
            e1 = e12.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e2 = e22.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            endo_pc = torch.cat([a12, a22, a32], dim=-1) * endo_pc
            endo_pc = scale2 * endo_pc
            endo_pc = endo_pc + e
            endo_pc = torch.bmm(endo_pc, torch.transpose(rotation_m2, 1, 2))
            endo_pc = endo_pc + trans2

            a13 = a13.repeat(1, 1, 60)  # N*50*50
            a14 = a14.repeat(1, 1, 40)  # N*50*20
            a134 = torch.cat([a13, a14], dim=-1)  # N*50*70
            a134 = a134.view(-1, Npoints_rv, 1)  # N*3500*1

            a23 = a23.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*3500*1
            a33 = a33.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*3500*1
            e3 = torch.zeros(e13.shape).cuda()
            e1 = e13.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*3500*1
            e2 = e23.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*3500*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*3500*1
            e = torch.cat([e1, e2, e3], dim=-1)
            rv = torch.cat([a134, a23, a33], dim=-1) * rv
            rv = scale3 * rv
            rv = rv + e
            rv = torch.bmm(rv, torch.transpose(rotation_m3, 1, 2))
            rv = rv + trans3

            loss111, _ = chamfer_distance(epi_pc, epi_myo_gt)
            loss112, _ = chamfer_distance(endo_pc, endo_myo_gt)
            loss11 = loss111 + loss112
            loss12, _ = chamfer_distance(rv, rv_gt)
            loss = loss11 + loss12 + sw * (sm_loss1 + sm_loss2 + sm_loss3)

        else:
            # print('NODE training')
            a1 = a1.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            a2 = a2.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            a3 = a3.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e3 = torch.zeros(e1.shape).cuda()
            e1 = e1.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e2 = e2.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_epi, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            rotation_m, _ = compute_rotation_matrix_from_quaternion(quaternion)  # Bx3x3
            epi_pc = torch.cat([a1, a2, a3], dim=-1) * epi_pc
            epi_pc = scale * epi_pc
            epi_pc = epi_pc + e
            epi_pc = torch.bmm(epi_pc, torch.transpose(rotation_m, 1, 2))
            epi_pc = epi_pc + trans

            a12 = a12.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            a22 = a22.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            a32 = a32.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e3 = torch.zeros(e12.shape).cuda()
            e1 = e12.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e2 = e22.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_endo, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            rotation_m2, _ = compute_rotation_matrix_from_quaternion(quaternion2)  # Bx3x3
            endo_pc = torch.cat([a12, a22, a32], dim=-1) * endo_pc
            endo_pc = scale2 * endo_pc
            endo_pc = endo_pc + e
            endo_pc = torch.bmm(endo_pc, torch.transpose(rotation_m2, 1, 2))
            endo_pc = endo_pc + trans2

            a13 = a13.repeat(1, 1, 60)  # N*50*35
            a14 = a14.repeat(1, 1, 40)  # N*50*35
            a134 = torch.cat([a13, a14], dim=-1)  # N*50*70
            a134 = a134.view(-1, Npoints_rv, 1)  # N*3500*1

            a23 = a23.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            a33 = a33.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            e3 = torch.zeros(e13.shape).cuda()
            e1 = e13.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            e2 = e23.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            e3 = e3.repeat(1, 1, Nr).view(-1, Npoints_rv, 1)  # N*5600*1
            e = torch.cat([e1, e2, e3], dim=-1)
            rotation_m3, _ = compute_rotation_matrix_from_quaternion(quaternion3)  # Bx3x3
            rv = torch.cat([a134, a23, a33], dim=-1) * rv
            rv = scale3 * rv
            rv = rv + e
            rv = torch.bmm(rv, torch.transpose(rotation_m3, 1, 2))
            rv = rv + trans3

            epi_pred0, epi_pred1 = GeoMorph.neural_mesh_forward1(code1.detach(), epi_pc.detach(), epi_myo_gt)
            endo_pred0, endo_pred1 = GeoMorph.neural_mesh_forward2(code2.detach(), endo_pc.detach(), endo_myo_gt)
            rv_pred0, rv_pred1 = GeoMorph.neural_mesh_forward3(code3.detach(), rv.detach(), rv_gt)

            pred0 = torch.cat([epi_pred0, endo_pred0, rv_pred0], dim=1)
            pred1 = torch.cat([epi_pred1, endo_pred1, rv_pred1], dim=1)

            loss111, _ = chamfer_distance(epi_pred0, epi_myo_gt)
            loss112, _ = chamfer_distance(endo_pred0, endo_myo_gt)
            loss11 = loss111 + loss112
            loss12, _ = chamfer_distance(rv_pred0, rv_gt)
            loss = loss11 + loss12 + \
                   0.1*(quaternion_loss(epi_pc.detach(), epi_pred0)) + \
                   0.1*(quaternion_loss(endo_pc.detach(), endo_pred0)) + \
                   0.1*(quaternion_loss(rv.detach(), rv_pred0)) + \
                   0.05 * displacement_smooothing_loss(epi_pc.detach(), epi_pred0) + \
                   0.05 * displacement_smooothing_loss(endo_pc.detach(), endo_pred0) + \
                   0.05 * displacement_smooothing_loss(rv.detach(), rv_pred0)


        loss_c2 = loss11 + loss12
        print('loss_chamfer')
        print(loss_c2)

        input_pc = points.squeeze(0)
        epi_target_pred = epi_pc.squeeze(0)
        endo_target_pred = endo_pc.squeeze(0)
        rv_target_pred = rv.squeeze(0)
        myo_gt = torch.cat([epi_myo_gt, endo_myo_gt], dim=1)

        pred0 = pred0.squeeze(0)
        pred1 = pred1.squeeze(0)
        myo_gt = myo_gt.squeeze(0)
        rv_gt = rv_gt.squeeze(0)

        print('input_pc.shape')
        print(input_pc.shape)

        print('myo_gt.shape')
        print(myo_gt.shape)
        print('rv_gt.shape')
        print(rv_gt.shape)

        print('epi_target_pred.shape')
        print(epi_target_pred.shape)
        print('endo_target_pred.shape')
        print(endo_target_pred.shape)

        data_root = os.path.join(model_path, 'mask_e' + str(epoch) + '_' + str(i))
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        epi_target_pred = epi_target_pred.cpu().detach().numpy()
        endo_target_pred = endo_target_pred.cpu().detach().numpy()
        rv_target_pred = rv_target_pred.cpu().detach().numpy()
        input_pc = input_pc.cpu().detach().numpy()
        pred0 = pred0.cpu().detach().numpy()
        pred1 = pred1.cpu().detach().numpy()

        data_root1 = os.path.join(data_root, 'input_pc.npz')
        np.savez_compressed(data_root1, input_pc)

        data_root1 = os.path.join(data_root, 'epi_mask_prediction.npz')
        np.savez_compressed(data_root1, epi_target_pred)

        data_root1 = os.path.join(data_root, 'endo_mask_prediction.npz')
        np.savez_compressed(data_root1, endo_target_pred)

        data_root1 = os.path.join(data_root, 'rv_mask_prediction.npz')
        np.savez_compressed(data_root1, rv_target_pred)

        data_root1 = os.path.join(data_root, 'pred0.npz')
        np.savez_compressed(data_root1, pred0)

        data_root1 = os.path.join(data_root, 'pred1.npz')
        np.savez_compressed(data_root1, pred1)

        myo_gt = myo_gt.cpu().detach().numpy()
        rv_gt = rv_gt.cpu().detach().numpy()

        data_root2 = os.path.join(data_root, 'myo_gt.npz')
        np.savez_compressed(data_root2, myo_gt)

        data_root2 = os.path.join(data_root, 'rv_gt.npz')
        np.savez_compressed(data_root2, rv_gt)


if __name__ == '__main__':
    main()

