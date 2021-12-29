import os
from os.path import join, basename, splitext, dirname, exists, abspath
import re
import sys
import time
import math
import glob
import argparse
import shutil

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import ezdxf
import dxfgrabber
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

from helper_ply import write_ply
from utils import print_args, save_args, random_color


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True, help='')
parser.add_argument('--format', type=str, default='txt', help='')
parser.add_argument('--voxel_size', type=float, default=0.01, help='')
parser.add_argument('--overlap_voxel_thres', type=int, default=1, help='')
parser.add_argument('--plane', type=str, default='PCA', help='A, B, C or PCA')
parser.add_argument('--projection', type=str, default='AB', help='A or AB')
parser.add_argument('--angle_thres', type=float, default=30.0, help='')
parser.add_argument('--average_normal', action="store_true", help='')
ARGS = parser.parse_args()


ROOT = ARGS.root
FORMAT = ARGS.format
VOXEL_SIZE = ARGS.voxel_size
OVERLAP_VOXEL_THRES = ARGS.overlap_voxel_thres
PLANE = ARGS.plane
PROJECTION = ARGS.projection
ANGLE_THRES = ARGS.angle_thres
AVERAGE_NORMAL = ARGS.average_normal

HALF_PI = math.pi / 2
Z_AXIS = np.array([0.0, 0.0, 1.0])


class Grain:
    def __init__(self, path, voxel_size=None, A_path: str=None, B_path: str=None, C_path: str=None) -> None:
        self.path = path
        if splitext(path)[-1] == '.npy':
            self.raw_data = np.load(f)
        else:
            self.raw_data = np.loadtxt(f)

        # voxelization
        if voxel_size is not None:
            self.data = voxelize(self.raw_data[:, 0:3], voxel_size)
            self.data = np.unique(self.data, axis=0)
        else:
            self.data = None

        self.A_path = A_path
        if self.A_path is not None and exists(self.A_path):
            self.A = self.read_dxf(self.A_path)
        else:
            self.A = None

        self.B_path = B_path
        if self.B_path is not None and exists(self.B_path):
            self.B = self.read_dxf(self.B_path)
        else:
            self.B = None

        self.C_path = C_path
        if self.C_path is not None and exists(self.C_path):
            self.C = self.read_dxf(self.C_path)
        else:
            self.C = None

        if self.A is not None and self.B is not None:
            self.AB = normalize(np.cross(self.A, self.B))
        elif self.C is not None:
            self.AB = normalize(self.C)
        else:
            self.AB = None

        if self.AB is not None and self.AB[2] < 0: self.AB *= -1

    def read_dxf(self, filepath):
        dxf = dxfgrabber.readfile(filepath)
        end_points = np.array(dxf.entities[0].points)
        axis = end_points[1] - end_points[0]

        if axis[2] < 0: # make z value positive
            return -1 * axis

        return axis


def voxelize(xyz, voxel_size):
    return (xyz / voxel_size).astype(np.int32)


def normalize(vector, eps=1e-7):
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector
    return vector / norm


def get_plane_normal(vector1, vector2):
    """Generate normal vector of 3D plane

    Args:
        vector1 (np.ndarray): A vector on the plane. Size = (3,).
        vector2 (np.ndarray): A vector on the plane. Size = (3,).
    """
    vector1_n = normalize(vector1)
    vector2_n = normalize(vector2)
    normal = np.cross(vector1_n, vector2_n)
    return normalize(normal)


def get_intersect_points(array1, array2):
    _, ncols = array1.shape
    dtype = {
        'names': ['f{}'.format(i) for i in range(ncols)],
        'formats': ncols * [array1.dtype]
    }

    intersect = np.intersect1d(array1.view(dtype), array2.view(dtype))
    intersect = intersect.view(array1.dtype).reshape(-1, ncols)

    return intersect


def get_included_angle(vector1, vector2, degree=False):
    rad = math.acos(
        np.dot(vector1, vector2) /
        (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

    if degree: return rad * 180.0 / math.pi

    return rad


def get_dihedral_angle(plane_normal1, plane_normal2, degree=True):
    rad = np.dot(plane_normal1, plane_normal2) / (np.linalg.norm(plane_normal1) * np.linalg.norm(plane_normal2))

    if degree: return rad * 180.0 / math.pi

    return rad


def project2vector(source, target):
    return np.dot(source, target) / float(np.sum(target**2)) * target


def project2plane(input_vector, plane_normal):
    v_n = project2vector(input_vector, plane_normal) # project to normal vector
    return input_vector - v_n


def intersectPlanes(plane_normal_1, plane_normal_2):
    return np.cross(plane_normal_1, plane_normal_2)


def pca(data):
    pca = PCA(n_components=3)
    pca.fit(data[:, 0:3])
    n_samples = data.shape[0]

    # We center the data and compute the sample covariance matrix.
    center = (np.min(data, axis=0) + np.max(data, axis=0)) / 2
    centered_data = data - center
    cov_matrix = np.dot(centered_data.T, centered_data) / n_samples

    # Get eigenvectors and eigenvalues
    eigen_values_pca = []
    eigen_vectors_pca = pca.components_
    for eigenvector in eigen_vectors_pca:
        eigen_values_pca.append(
            np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))

    eigen_values_pca = np.array(eigen_values_pca)
    eigen_vectors_pca = np.array(eigen_vectors_pca)

    # Sort descending with eigenvalues
    order = np.argsort(-eigen_values_pca)
    eigen_values_pca = eigen_values_pca[order]
    eigen_vectors_pca = eigen_vectors_pca[order]

    return eigen_values_pca, eigen_vectors_pca


def get_polygon_dxf(plane_normal, center, size=(0.5,0.5,0.5)):
    direction = np.cross(plane_normal, Z_AXIS)
    w, _, d = size

    pt1 = center + direction * w * 0.5
    pt2 = center - direction * w * 0.5

    pts = [(pt1[0], pt1[1], pt1[2] + d * 0.5),
           (pt1[0], pt1[1], pt1[2] - d * 0.5),
           (pt2[0], pt2[1], pt2[2] - d * 0.5),
           (pt2[0], pt2[1], pt2[2] + d * 0.5),
           (pt1[0], pt1[1], pt1[2] + d * 0.5)]

    doc = ezdxf.new('R2000')
    msp = doc.modelspace()
    msp.add_polyline3d(pts)

    return doc


def region_query(source_grain,
                 compare_grains,
                 visited=None,
                 cross_vector=np.array([0.0, 0.0, 1.0]),
                 plane='PCA',
                 projection='AB',
                 angle_thres=30,
                 average_normal=False):
    nbr_indices = []
    dxf = []

    if plane == 'A' or plane == 'a':
        source_normal = get_plane_normal(cross_vector, source_grain.A)
    elif plane == 'B' or plane == 'b':
        source_normal = get_plane_normal(cross_vector, source_grain.B)
    elif plane == 'C' or plane == 'c':
        source_normal = get_plane_normal(cross_vector, source_grain.C)
    elif plane == 'PCA' or plane == 'pca':
        source_normal = None
    else:
        print("Plane axis {} is not support, use A-axis".format(plane))
        source_normal = get_plane_normal(cross_vector, source_grain.A)

    normal = source_normal

    for j, compare_grain in enumerate(compare_grains):
        if visited is not None and visited[j]: continue

        intersect = get_intersect_points(source_grain.data, compare_grain.data)
        if len(intersect) < OVERLAP_VOXEL_THRES: continue

        # check a-axes are same direction in XY plane
        # source_axis_xy = source_axis[0:2] if source_axis[2] > 0 else -1 * source_axis[0:2]
        # axis_xy = axis[0:2] if axis[2] > 0 else -1 * axis[0:2]
        # if np.sum(source_axis_xy * axis_xy) < 0: continue

        # form plane
        if plane == 'PCA' or plane == 'pca':
            merged_data = np.concatenate([source_grain.raw_data[:,0:3], compare_grain.raw_data[:,0:3]], axis=0)
            eigen_values_pca, eigen_vectors_pca = pca(merged_data)
            normal = get_plane_normal(cross_vector, eigen_vectors_pca[0,:]) # cross z-axis and eigenvector with largest principal component

            min_pt = np.min(merged_data[:, 0:3], axis=0)
            max_pt = np.max(merged_data[:, 0:3], axis=0)
            center = (min_pt + max_pt) / 2

            dxf.append(get_polygon_dxf(normal, center, max_pt-min_pt))
            dxf[0].saveas("pca.dxf")

        elif plane == 'CENTER' or plane == 'center':
            source_grain_centroid = (source_grain.raw_data[:, 0:3].max(axis=0) + source_grain.raw_data[:, 0:3].min(axis=0)) / 2
            compare_grain_centroid = (compare_grain.raw_data[:, 0:3].max(axis=0) + compare_grain.raw_data[:, 0:3].min(axis=0)) / 2
            normal = get_plane_normal(cross_vector, source_grain_centroid - compare_grain_centroid)

            dxf.append(get_polygon_dxf(normal, (source_grain_centroid+compare_grain_centroid)/2))
            dxf[0].saveas("center.dxf")

        elif plane == 'NONE' or plane == 'none':
            normal = None
        elif average_normal:
            if plane == 'A':
                compare_normal = get_plane_normal(cross_vector, compare_grain.A)
            elif plane == 'B':
                compare_normal = get_plane_normal(cross_vector, compare_grain.B)
            else:
                compare_normal = get_plane_normal(cross_vector, compare_grain.C)

            if np.dot(source_normal, compare_normal) > 0:
                normal = normalize((source_normal + compare_normal) / 2)
            else:
                normal = normalize((source_normal - compare_normal) / 2)

        # projection
        if projection == "A" or projection == "a":
            projected_source_axis = normalize(project2plane(source_grain.A, normal))
            projected_axis = normalize(project2plane(compare_grain.A, normal))

        elif projection == "AB" or projection == "ab":
            projected_source_axis = normalize(intersectPlanes(source_grain.AB, normal))
            projected_axis = normalize(intersectPlanes(compare_grain.AB, normal))

            if projected_source_axis[2] < 0: projected_source_axis *= -1
            if projected_axis[2] < 0: projected_axis *= -1

        else:
            projected_source_axis = None
            projected_axis = None

        # calculate included angle
        if plane == 'NONE' or plane == 'none':
            angle = get_dihedral_angle(source_grain.AB, compare_grain.AB, True)
        else:
            angle = get_included_angle(projected_axis, projected_source_axis, True)

        if angle <= angle_thres:
            nbr_indices.append(j)

    return nbr_indices


def clustering(grains,
               cross_vector=np.array([0.0, 0.0, 1.0]),
               plane='PCA',
               projection='AB',
               angle_thres=30,
               average_normal=False):
    clusters = []
    clustered = np.full((len(grains),), False, dtype=np.bool)
    visited = np.full((len(grains),), False, dtype=np.bool)

    cluster_id = 0
    for i, grain in tqdm(enumerate(grains)):
        if visited[i] == False:
            visited[i] = True

            nbr_indices = region_query(grain, grains, visited, cross_vector, plane, projection, angle_thres, average_normal)

            clustered[i] = True
            clusters.append([i])

            for n_idx in nbr_indices:
                if visited[n_idx] == False:
                    visited[n_idx] = True

                    nbr_indices2 = region_query(grains[n_idx], grains, visited, cross_vector, plane, projection, angle_thres, average_normal)
                    nbr_indices.extend(nbr_indices2)

                if clustered[n_idx] == False:
                    clustered[n_idx] = True
                    clusters[cluster_id].append(n_idx)

            cluster_id += 1

    return clusters


if __name__ == "__main__":
    print_args(ARGS)

    # Get files
    files = sorted(glob.glob(join(ROOT, "*." + FORMAT)))
    print("Number of files:", len(files))

    # Load grains
    grains = []
    for f in tqdm(files):
        file_dir = dirname(f)
        file_basename_wo_ext = splitext(basename(f))[0]

        # axes
        a_axis_path = join(file_dir, file_basename_wo_ext + "_A.dxf")
        b_axis_path = join(file_dir, file_basename_wo_ext + "_B.dxf")
        c_axis_path = join(file_dir, file_basename_wo_ext + "_C.dxf")

        # Create grain
        grains.append(Grain(f, VOXEL_SIZE, a_axis_path, b_axis_path, c_axis_path))

    # Clustering
    s = time.time()
    clusters = clustering(grains, Z_AXIS, PLANE, PROJECTION, ANGLE_THRES, AVERAGE_NORMAL)
    e = time.time()
    print(e - s)

    # Save results
    cluster_colors = random_color(len(clusters))
    output_log_name = join(dirname(ROOT), "imbrication_{}_{}_{}_{}_{}".format(VOXEL_SIZE, PLANE, PROJECTION, ANGLE_THRES, AVERAGE_NORMAL))
    cluster_output_dir = output_log_name
    save_args(output_log_name + ".txt", ARGS)

    if not exists(cluster_output_dir):
        os.makedirs(cluster_output_dir)

    fout = open(output_log_name + ".csv", "w")
    for i, (cluster, color) in enumerate(zip(clusters, cluster_colors)):
        print(cluster)

        cluster_name = str(i).zfill(4)
        if not exists(join(cluster_output_dir, cluster_name)):
            os.makedirs(join(cluster_output_dir, cluster_name))

        cluster_data = []
        for grain_idx in cluster:
            cluster_data.append(grains[grain_idx].raw_data[:, 0:3])
            shutil.copyfile(grains[grain_idx].A_path, join(cluster_output_dir, cluster_name, basename(grains[grain_idx].A_path)))
            write_ply(join(cluster_output_dir, cluster_name, splitext(basename(grains[grain_idx].path))[0] + ".ply"),
                [grains[grain_idx].raw_data[:,0:3], grains[grain_idx].raw_data[:,3:].astype(np.uint8)],
                ['x', 'y', 'z', 'red', 'green', 'blue'])
            # shutil.copyfile(grains[grain_idx].path, join(cluster_output_dir, cluster_name, basename(grains[grain_idx].path)))

            fout.write(str(grain_idx) + ",")
        fout.write("\n")

        cluster_data = np.concatenate(cluster_data, axis=0)
        cluster_color = np.tile(color.reshape(1, -1), [len(cluster_data), 1])
        write_ply(join(cluster_output_dir, cluster_name + ".ply"), [cluster_data, cluster_color], ['x', 'y', 'z', 'red', 'green', 'blue'])

    fout.close()