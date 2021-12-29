import os
import sys
import glob
import argparse
from time import time
from os.path import join, splitext, basename, dirname, abspath, exists

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np
from helper_ply import read_ply, write_ply
from sklearn.decomposition import PCA
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, required=True, help='point cloud of scene (*.ply)')
parser.add_argument('--filedir', type=str, required=True, help='path to gravel point cloud')
parser.add_argument('--format', type=str, default='txt', help='')
parser.add_argument('--outdir', type=str, required=True, help='output folder path')
ARGS = parser.parse_args()

def coordinate_transform(T, xyz):
    xyz_4d = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=-1).T
    return (np.dot(T, xyz_4d).T)[:, 0:3]


def pca_transformation(data):
    pca = PCA(n_components=3)
    pca.fit(data[:, 0:3])
    n_samples = data.shape[0]

    # We center the data and compute the sample covariance matrix.
    center = (np.min(data, axis=0) + np.max(data, axis=0)) / 2
    centered_data = data - center
    cov_matrix = np.dot(centered_data.T, centered_data) / n_samples

    # get eigenvectors and eigenvalues
    eigen_values_pca = []
    eigen_vectors_pca = pca.components_
    for eigenvector in eigen_vectors_pca:
        eigen_values_pca.append(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))

    eigen_values_pca = np.array(eigen_values_pca)
    eigen_vectors_pca = np.array(eigen_vectors_pca)

    # sort descending with eigenvalues
    order = np.argsort(-eigen_values_pca)
    eigen_vectors_pca = eigen_vectors_pca[order]
    x_axis = eigen_vectors_pca[0, :]
    y_axis = eigen_vectors_pca[1, :]
    z_axis = np.cross(x_axis, y_axis)

    # check x, z axis
    if np.sum(np.array([0, 0, 1]) * z_axis) < 0: z_axis *= -1
    if np.sum(np.array([1, 0, 0]) * x_axis) < 0: x_axis *= -1
    y_axis = np.cross(z_axis, x_axis)

    # transform
    T = np.eye(4)
    T[0, 0:3] = x_axis
    T[1, 0:3] = y_axis
    T[2, 0:3] = z_axis
    T[0:3, 3:4] = -1 * np.dot(T[0:3, 0:3], np.reshape(center, [-1, 1]))
    data_transformed = coordinate_transform(T, data)

    return data_transformed, T


if __name__ == "__main__":
    filedir = ARGS.filedir
    files = sorted(glob.glob(join(filedir, "*."+ARGS.format)))
    outdir = ARGS.outdir
    filename = ARGS.filename
    print("Processing {}...".format(filename))

    ply_data = read_ply(filename)
    data = np.vstack((ply_data['x'], ply_data['y'], ply_data['z'],
                      ply_data['red'], ply_data['green'], ply_data['blue'])).T
    print("Data shape:", data.shape)

    xyz = data[:, 0:3]
    start_time = time()
    xyz_pca, T = pca_transformation(xyz)
    end_time = time()
    print("Processing time:", end_time - start_time)

    output_filename = join(dirname(filename), splitext(basename(filename))[0] + "_pca.ply")
    colors = data[:, 3:].astype(np.uint8)
    output_data = [xyz_pca, colors]
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    write_ply(output_filename, output_data, field_names)
    print("Saved transformed data to", output_filename)

    if not exists(outdir):
        os.makedirs(outdir)

    for f in tqdm(files):
        grain = np.loadtxt(f)
        grain_colors = grain[:, 3:].astype(np.uint8)
        grain_pca = coordinate_transform(T, grain[:,0:3])

        output_grain_path = join(outdir, splitext(basename(f))[0] + "_pca."+ARGS.format)
        with open(output_grain_path, "w") as fout:
            for xyz, rgb in zip(grain_pca, grain_colors):
                line = "{:.8f} {:.8f} {:.8f} {} {} {}\n".format(
                    xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2])
                fout.write(line)
        # np.savetxt(join(outdir,
        #                 splitext(basename(f))[0] + "_pca.asc"),
        #            np.concatenate([grain_pca, grain_colors], axis=-1))
