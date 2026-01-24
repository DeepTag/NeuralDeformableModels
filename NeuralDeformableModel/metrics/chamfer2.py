#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh

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


def compute_trimesh_chamfer(gt_points_np, gen_mesh, num_mesh_samples=3000, FPS=True, offset=0, scale=1):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

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

    # only need numpy array of points
    # gt_points_np = gt_points.vertices

    # print('gen_points_sampled.shape')
    # print(gen_points_sampled.shape)
    # print('gt_points_np.shape')
    # print(gt_points_np.shape)


    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return {'chamfer_distance': gt_to_gen_chamfer + gen_to_gt_chamfer}
