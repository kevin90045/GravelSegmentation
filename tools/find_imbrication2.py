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
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

from helper_ply import write_ply
from utils import print_args, save_args, random_color


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True, help='')
parser.add_argument('--format', type=str, default='txt', help='')
parser.add_argument('--iou_thres', type=float, default=0.01, help='')
ARGS = parser.parse_args()

ROOT = ARGS.root
FORMAT = ARGS.format
IOU_THRES = ARGS.iou_thres


class Grain:
    def __init__(self, path) -> None:
        self.path = path
        if splitext(path)[-1] == '.npy':
            self.data = np.load(f)
        else:
            self.data = np.loadtxt(f)

        # convex hull in 2D
        hull = ConvexHull(self.data[:, 0:2])
        vertices_ind = list(hull.vertices) + [hull.vertices[0]]
        self.convex_hull_vertices = self.data[vertices_ind, 0:2]
        self.convex_hull = Polygon([[p[0], p[1]] for p in self.convex_hull_vertices])
        self.area = self.convex_hull.area


def intersect(grain1: Grain, grain2: Grain):
    return grain1.convex_hull.intersection(grain2.convex_hull).area


def region_query(source_grain, compare_grains, visited=None, iou_thres=0.01):
    nbr_indices = []

    for j, compare_grain in enumerate(compare_grains):
        if visited is not None and visited[j]: continue

        inter_area = intersect(source_grain, compare_grain)
        source_ratio = inter_area / source_grain.area
        compare_ratio = inter_area / compare_grain.area
        iou = inter_area / (source_grain.area + compare_grain.area - inter_area)

        # if iou < iou_thres or source_ratio >= 0.9 or compare_ratio >= 0.9:
        #     continue
        if max(source_ratio, compare_ratio) < iou_thres or source_ratio >= (1-iou_thres) or compare_ratio >= (1-iou_thres):
            continue

        nbr_indices.append(j)

    return nbr_indices


def clustering(grains, iou_thres=0.01):
    clusters = []
    clustered = np.full((len(grains),), False, dtype=np.bool)
    visited = np.full((len(grains),), False, dtype=np.bool)

    cluster_id = 0
    for i, grain in tqdm(enumerate(grains)):
        if visited[i] == False:
            visited[i] = True

            nbr_indices = region_query(grain, grains, visited, iou_thres)

            clustered[i] = True
            clusters.append([i])

            for n_idx in nbr_indices:
                if visited[n_idx] == False:
                    visited[n_idx] = True

                    nbr_indices2 = region_query(grains[n_idx], grains, visited, iou_thres)
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
        grains.append(Grain(f))

    for gi in grains:
        # hull_path = splitext(gi.path)[0] + "_hull.txt"
        # np.savetxt(hull_path, gi.convex_hull_vertices)

        dxf_path = splitext(gi.path)[0] + "_hull.dxf"
        doc = ezdxf.new('R2000')
        msp = doc.modelspace()
        msp.add_polyline2d(gi.convex_hull_vertices)
        doc.saveas(dxf_path)
        # for gj in grains:
        #     inter_area = intersect(gi, gj)
        #     source_ratio = inter_area / gi.area
        #     compare_ratio = inter_area / gj.area
        #     print('{:3.3f} '.format(source_ratio * 100.0), end='')
        # print()

    # Clustering
    s = time.time()
    clusters = clustering(grains, IOU_THRES)
    e = time.time()
    print(e - s)

    # Save results
    cluster_colors = random_color(len(clusters))
    output_log_name = join(dirname(ROOT), "imbrication2_{}".format(IOU_THRES))
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
            cluster_data.append(grains[grain_idx].data[:, 0:3])
            write_ply(join(cluster_output_dir, cluster_name, splitext(basename(grains[grain_idx].path))[0] + ".ply"),
                [grains[grain_idx].data[:,0:3], grains[grain_idx].data[:,3:].astype(np.uint8)],
                ['x', 'y', 'z', 'red', 'green', 'blue'])
            shutil.copyfile(join(ROOT, splitext(basename(grains[grain_idx].path))[0] + "_hull.dxf"),
                join(cluster_output_dir, cluster_name, splitext(basename(grains[grain_idx].path))[0] + "_hull.dxf"))

            fout.write(str(grain_idx) + ",")
        fout.write("\n")

        cluster_data = np.concatenate(cluster_data, axis=0)
        cluster_color = np.tile(color.reshape(1, -1), [len(cluster_data), 1])
        write_ply(join(cluster_output_dir, cluster_name + ".ply"), [cluster_data, cluster_color], ['x', 'y', 'z', 'red', 'green', 'blue'])

        command_line = 'CloudCompare -SILENT'
        for grain_idx in cluster:
            command_line += ' -O ' + join(dirname(grains[grain_idx].path), splitext(basename(grains[grain_idx].path))[0] + ".asc")
        command_line += ' -SAVE_CLOUDS ALL_AT_ONCE'
        os.system(command_line)

    fout.close()