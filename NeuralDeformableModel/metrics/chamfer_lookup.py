#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


def pc_morph(gt_points_np, gen_points_np, gen_points_es_np):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    # one direction
    gen_points_kd_tree = KDTree(gen_points_np)

    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)


    morphed_points = gen_points_es_np[one_vertex_ids]

    # print('gt_points_ed_np.shape:', gt_points_np.shape)
    # print('gen_points_ed_np.shape:', gen_points_np.shape)
    # print('gen_points_es_np.shape:', gen_points_es_np.shape)
    # print('morphed_points.shape:', morphed_points.shape)

    return morphed_points
