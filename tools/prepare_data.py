import os
import sys
import glob
import argparse
from datetime import datetime
from functools import partial
import multiprocessing as mp
from concurrent import futures

import h5py
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from helper_data import DataUtils
from utils import print_args


parser = argparse.ArgumentParser()
parser.add_argument("--root", help="path to root directory of scenes")
parser.add_argument("--dir", help="files in the same directory will be merge as one scene", action="store_true")
parser.add_argument("--seed", type=int, default=42, help="random number seed")
parser.add_argument("--num_points", type=int, default=8192, help="point number of each block")
parser.add_argument("--block_size", type=float, default=1.0, help="block size")
parser.add_argument("--stride", type=float, default=0.5, help="stride between blocks")
parser.add_argument("--bound", nargs="+", type=float, default=[-sys.float_info.max, -sys.float_info.max, sys.float_info.max, sys.float_info.max], help="clip boundary of input points: xmin ymin xmax ymax")
parser.add_argument("--format", type=str, default="h5", help="data format")
parser.add_argument("--idis", help="use inverse density importance sampling", action="store_true")
parser.add_argument("-mp", "--multiprocessing", help="use multiprocessing or not", action="store_true")
parser.add_argument("--num_workers", type=int, default=1, help="threads used for multiprocessing")
parser.add_argument("--start_index", type=int, default=0, help="the startig index of output file")
parser.add_argument("--save_original", help="save original scene data as h5", action="store_true")
args = parser.parse_args()

# arguments
ROOT = args.root
LOAD_DIR = args.dir
SEED = args.seed
NUM_POINT = args.num_points
BLOCK_SIZE = args.block_size
STRIDE = args.stride
XMIN, YMIN, XMAX, YMAX = args.bound
FORMAT = args.format
USE_IDIS = args.idis
USE_MULTIPROCESSING = args.multiprocessing
NUM_WORKERS = args.num_workers
START_INDEX = args.start_index
SAVE_ORIGINAL = args.save_original

class Logger:
    """Log text to storage
    """
    def __init__(self, output_filename=None) -> None:
        """Create Logger

        Args:
            output_filename (str, optional): Filename of log file. If not specified, use timestamp. Defaults to None.
        """
        self.output_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if output_filename is None else output_filename
        self.log = open(self.output_filename + ".txt", "w")

    def write(self, log_str):
        """Write log

        Args:
            log_str (str): String to be logged
        """
        self.log.write(log_str + "\n")
        self.log.flush()

    def close(self):
        """Close log file
        """
        self.log.close()


def print_log(log_str):
    """Print log with timestamp

    Args:
        log_str (str): The string to be printed
    """
    print("{} - {}".format(datetime.now(), log_str))


def scene_to_blocks(scene_data, num_points, size=1.0, stride=0.5, threshold=100, use_idis=False, use_multiprocessing=False, num_workers=1):
    """Convert scene points to blocks

    Args:
        scene_data (numpy.ndarray): Nx4: x, y, z and instance label
        num_points (int): Number of points in each block
        size (float, optional): Block size. Defaults to 1.0.
        stride (float, optional): Stride between blocks. Defaults to 0.5.
        threshold (int, optional): Minimum number of points for each instance. If < theshold, the instance will be eliminated. Defaults to 100.
        use_idis (bool, optional): If True, use Inverse Density Importance Sampling. Defaults to False.
        pool (multiprocessing.Pool, optional): An instance of multiprocessing.Pool. If passed, use multiproessing when sampling. Defaults to None.

    Returns:
        numpy.ndarray: Nx9: block id, num_points, global coordinates, block centered coordinates, room normalized coordinates, instance labels
    """
    from tqdm import tqdm
    scene_data[:, 0:3] = scene_data[:, 0:3] - np.amin(scene_data[:, 0:3], axis=0, keepdims=True)

    limit = np.amax(scene_data[:, 0:3], axis=0)
    width = int(np.ceil((limit[0] - size) / stride)) + 1
    depth = int(np.ceil((limit[1] - size) / stride)) + 1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = []
    for (x, y) in tqdm(cells):
        xcond = (scene_data[:, 0] <= x + size) & (scene_data[:, 0] >= x)
        ycond = (scene_data[:, 1] <= y + size) & (scene_data[:, 1] >= y)
        cond = xcond & ycond

        if np.sum(cond) < threshold:
            continue

        block = scene_data[cond, :]

        if block.shape[0] is 0:
            continue
        blocks.append(block)

    if use_multiprocessing:
        chunksize = len(blocks) // num_workers
        with futures.ProcessPoolExecutor(num_workers) as pool:
            blocks = list(pool.map(partial(DataUtils.sample_data, num_samples=num_points, min_instance_points_num=50, use_idis=use_idis), blocks, chunksize=chunksize))
    else:
        for block_id, block in enumerate(blocks):
            blocks[block_id] = DataUtils.sample_data(block, num_points, min_instance_points_num=50, use_idis=use_idis)

    blocks = np.stack(blocks, axis=0)
    # A batch should have shape of BxNx14, where
    # [0:3] - global coordinates
    # [3:6] - block centered coordinates (centered at Z-axis)
    # [6:9] - room normalized coordinates
    # [9] - instance labels
    num_blocks = blocks.shape[0]
    batch = np.zeros((num_blocks, num_points, 14))
    for b in range(num_blocks):
        minx = min(blocks[b, :, 0])
        miny = min(blocks[b, :, 1])
        batch[b, :, 3] = blocks[b, :, 0] - (minx + size * 0.5)
        batch[b, :, 4] = blocks[b, :, 1] - (miny + size * 0.5)
        batch[b, :, 6] = blocks[b, :, 0] / limit[0]
        batch[b, :, 7] = blocks[b, :, 1] / limit[1]
        batch[b, :, 8] = blocks[b, :, 2] / limit[2]
    batch[:, :, 0:3] = blocks[:, :, 0:3]
    batch[:, :, 5] = blocks[:, :, 2]
    batch[:, :, 9] = blocks[:, :, 3]
    return batch


def save_batch_h5(fname, batch):
    """Save batch (blocks) to HDF5 file

    Args:
        fname (str): output filename
        batch (numpy.ndarray): Nx9: block id, num_points, global coordinates, block centered coordinates, room normalized coordinates, instance labels
    """
    fp = h5py.File(fname, "w")
    coords = batch[:, :, 0:3]
    points = batch[:, :, 3:9]
    labels = batch[:, :, 9]
    fp.create_dataset("coords", data=coords, compression="gzip", dtype="float32", compression_opts=9)
    fp.create_dataset("points", data=points, compression="gzip", dtype="float32", compression_opts=9)
    fp.create_dataset("labels", data=labels, compression="gzip", dtype="int64", compression_opts=9)
    fp.close()


def save_scene_h5(fname, scene_data):
    """Save scene data to HDF5 file

    Args:
        fname (str): output filename
        scene_data (numpy.ndarray): Nx4: x, y, z and instance label
    """
    fp = h5py.File(fname, "w")
    coords = scene_data[:, 0:3]
    labels = scene_data[:, 3]
    fp.create_dataset("coords", data=coords, compression="gzip", dtype="float32", compression_opts=9)
    fp.create_dataset("labels", data=labels, compression="gzip", dtype="int64", compression_opts=9)
    fp.close()


if __name__ == "__main__":
    print_args(args)
    np.random.seed(SEED) # seed for sample data randomly

    # get all scene directories/fiiepaths
    all_scene_paths = []
    for dir_path, _, _ in os.walk(ROOT):
        if LOAD_DIR:
            all_scene_paths.append(dir_path)
        else:
            scene_paths = glob.glob(os.path.join(dir_path, "*." + FORMAT))
            all_scene_paths.extend(scene_paths)
    all_scene_paths.sort()

    # logger
    scene_count = START_INDEX
    logger = Logger()

    # start processing
    failed_scenes = []
    for i, scene_path in enumerate(all_scene_paths):
        print("[{}/{}]".format(i+1, len(all_scene_paths)))
        # load scene data
        try:
            if LOAD_DIR:
                scene_data = DataUtils.load_scene_dir(scene_path, FORMAT, xmin=XMIN, ymin=YMIN, xmax=XMAX, ymax=YMAX)
            else:
                scene_data = DataUtils.load_scene_file(scene_path, FORMAT, xmin=XMIN, ymin=YMIN, xmax=XMAX, ymax=YMAX)
        except:
            print_log("Unexpected error: {}".format(sys.exc_info()[0]))
            print_log("Failed to load {}, Skip\n".format(scene_path))
            failed_scenes.append(scene_path)
            continue

        if scene_data.shape[0] > 0:
            print_log("Number of total points: {}".format(scene_data.shape[0]))

            # scene to blocks
            blocks = scene_to_blocks(scene_data, NUM_POINT, BLOCK_SIZE, STRIDE, 1, USE_IDIS, USE_MULTIPROCESSING, NUM_WORKERS)
            print_log("Number of points after sampling: {}".format(blocks.shape[0] * blocks.shape[1]))

            # output
            if LOAD_DIR:
                output_basename = os.path.join(scene_path, "scene_" + str(scene_count).zfill(4))
            else:
                output_basename = os.path.join(os.path.dirname(scene_path), "scene_" + str(scene_count).zfill(4))

            output_batch_filename = output_basename + ".h5"
            print_log("Saving batch to {}...".format(output_batch_filename))
            save_batch_h5(output_batch_filename, blocks)

            if SAVE_ORIGINAL:
                output_scene_filename = output_basename + "_original.h5"
                print_log("Saving scene to {}...\n".format(output_scene_filename))
                save_scene_h5(output_scene_filename, scene_data)

            # logging
            scene_count = scene_count + 1
            logger.write(output_batch_filename)

    print_log("Finished {} scenes".format(scene_count - START_INDEX))

    # display failed scenes
    if len(failed_scenes) > 0:
        print("\nFailed scenes:")
        logger.write("\nFailed Scenes:")

    for failed_scene_filename in failed_scenes:
        print(failed_scene_filename)
        logger.write(failed_scene_filename)

    print_log("Done")
    logger.close()
