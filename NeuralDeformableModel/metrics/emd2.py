#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import trimesh.sample
# from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

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


def compute_trimesh_emd(gt_points_np, gen_mesh, num_mesh_samples=3000, FPS=True, offset=0, scale=1):
    """
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    if FPS:
        c_epi_1 = farthest_point_sample(gen_mesh, npoint=num_mesh_samples)
        gen_points_sampled = gen_mesh[c_epi_1]
    else:
        gen_points_sampled = gen_mesh

    gen_points_sampled = gen_points_sampled / scale - offset

    gt_points_np = np.random.permutation(gt_points_np)[:num_mesh_samples]

    dist = np.linalg.norm(np.expand_dims(gt_points_np, axis=0) - np.expand_dims(gen_points_sampled, axis=1), axis=-1)

    assignment = linear_sum_assignment(dist)

    emd = dist[assignment].sum() / num_mesh_samples



    # print('emd:', emd)
    # print('emd2:', emd2)


    return {'earthmover_distance': emd}
