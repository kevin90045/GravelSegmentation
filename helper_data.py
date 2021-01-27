import os
import gc
import sys
import glob
import math
import multiprocessing
from concurrent import futures

import h5py
import numpy as np
import numpy
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class DataConfigs:
    """Data configurations
    """
    # number of channels, default: 6 (x, y, z, scene normalized x, scene normalized y, scene normalized z)
    points_cc = 6

    # Maximum number of output instances
    ins_max_num = 60

    # Number of points in each block when training
    train_pts_num = 8192

    # Number of points in each block when testing
    test_pts_num = 8192


class DataLoader:
    """Data loader for getting batch data
    
    Modified from: https://github.com/dlinzhao/JSNet/blob/master/utils/s3dis_utils/dataset_s3dis.py
    """
    def __init__(self,
                 dataset_path=".",
                 sub_dirs=None,
                 filepaths=None,
                 configs=DataConfigs,
                 epoch=1,
                 batch_size=4,
                 num_works=1,
                 shuffle=False):
        """Create data

        Args:
            dataset_path (str, optional): Root path of dataset. Defaults to ".".
            sub_dirs (list, optional): List of sub-directories. Used when filepaths is None. Defaults to None.
            filepaths (list, optional): List of file paths. If provide, dataset_path and sub_dirs will be ignored. Defaults to None.
            configs (DataConfigs, optional): Data configurations. Defaults to DataConfigs.
            epoch (int, optional): Number of epochs which is at least 1. Defaults to 1.
            batch_size (int, optional): Batch size. Defaults to 4.
            num_works (int, optional): Number of workers for multiprocessing data pipeline which is at least 1. Defaults to 1.
            shuffle (bool, optional): Shuffle scenes every epoch. Defaults to False.
        """
        self.root_folder = dataset_path if filepaths is None else None
        self.filepaths = filepaths
        self.files, self.length = self.load_file_list(sub_dirs=sub_dirs)
        print('files: {}, blocks: {}'.format(len(self.files), self.length))

        self.ins_max_num = configs.ins_max_num
        self.batch_size = batch_size
        self.total_batch_num = DataLoader.get_total_batch_num(self.length, self.batch_size)        

        # for data pipeline
        self.epoch = epoch
        self.num_works = num_works
        self.shuffle = shuffle

    def __del__(self):
        try:
            self.consumer_process.terminate()
            self.producer_process.terminate()
        except:
            pass

    def __len__(self):
        return len(self.length)
    
    def init_data_pipeline(self):
        """Initialize data pipeline. Must be called before calling get_batch()
        """
        self.total_batch_num = DataLoader.get_total_batch_num(self.length, self.batch_size)
                
        self.capacity = 30

        self.data_sample_queue = multiprocessing.Manager().Queue(3) # max size: 3 scenes
        self.data_queue = multiprocessing.Manager().Queue(self.capacity) # max size: 30 batches

        self.producer_process = multiprocessing.Process(target=DataLoader.data_sample, args=(
            self.data_sample_queue, self.files, self.ins_max_num, self.epoch, self.num_works, self.shuffle))

        self.consumer_process = multiprocessing.Process(target=DataLoader.data_prepare, args=(
            self.data_sample_queue, self.data_queue, self.length, self.epoch, self.batch_size))

        self.producer_process.start()
        self.consumer_process.start()

    def load_file_list(self, sub_dirs: list) -> tuple:
        """Load file list: HDF5 paths, total number of blocks

        Args:
            sub_dirs (list): List of sub-folders under self.root_folder. Used when self.filepaths is not None.

        Returns:
            tuple (list, int): Tuple of list of HDF5 paths and total number of blocks
        """
        # get paths of all HDF5 files
        all_files = []
        if self.filepaths is None:
            if sub_dirs is None:
                all_files.extend(sorted(glob.glob(os.path.join(self.root_folder, '*.h5'))))
            else:
                for dir in sub_dirs:
                    all_files.extend(sorted(glob.glob(os.path.join(self.root_folder, dir, '*.h5'))))
        else:
            all_files.extend(sorted(self.filepaths))
        
        # compute number of all blocks
        length = 0
        for file in tqdm(all_files):
            fin = h5py.File(file, 'r')
            block_num = fin['labels'][:].shape[0]
            length += block_num
            fin.close()

        return all_files, length

    @staticmethod
    def get_total_batch_num(num_data: int, batch_size: int) -> int:
        """Get total batch number

        Args:
            num_data (int): Number of all data (blocks).
            batch_size (int): Batch size.

        Returns:
            int: Total batch number.
        """
        return int(math.ceil(num_data / batch_size))

    @staticmethod
    def get_bbvert_pmask_labels(pc, ins_labels, ins_max_num) -> tuple:
        """Get bounding box vertices and masks from points and labels

        Args:
            pc (numpy.ndarray): NxC, the first three channels should be x, y, z.
            ins_labels (numpy.ndarray): N instances labels.
            ins_max_num (int): Maximum number of instances.

        Returns:
            tuple (numpy.ndarray, numpy.ndarray): bounding box vertices (ins_max_num, 2, 3) and masks (ins_max_num, N).
        """

        # initialize bounding boxes and masks
        gt_bbvert_padded = np.zeros((ins_max_num, 2, 3), dtype=np.float32)
        gt_pmask = np.zeros((ins_max_num, pc.shape[0]), dtype=np.float32)        
        
        # get unique instance labels, point indices of each instance, point number of each instance
        unique_ins_labels, pt_ins_ind, ins_pt_counts = np.unique(ins_labels, return_inverse=True, return_counts=True, axis=0)
        ins_pt_ind = np.split(np.argsort(pt_ins_ind), np.cumsum(ins_pt_counts[:-1]))
        
        for count, (ins_ind, pt_ind) in enumerate(zip(unique_ins_labels, ins_pt_ind)):
            # only consider label >= 0
            if ins_ind <= -1:
                continue

            # check if the current number of instances is larger than ins_max_num
            if count >= ins_max_num:
                print('Ignored! more than max instances:', len(unique_ins_labels))
                continue
            
            # create one-hot point mask
            ins_labels_tp = np.zeros(ins_labels.shape, dtype=np.int8).reshape(-1)
            ins_labels_tp[pt_ind] = 1
            gt_pmask[count, :] = ins_labels_tp

            # create bounding box vertices: min_xyz, max_xyz
            pc_xyz_tp = pc[pt_ind, 0:3]
            gt_bbvert_padded[count, 0, :] = np.min(pc_xyz_tp, axis=0)
            gt_bbvert_padded[count, 1, :] = np.max(pc_xyz_tp, axis=0)

        return gt_bbvert_padded, gt_pmask

    @staticmethod
    def load_fixed_points(file_path: str, ins_max_num: int) -> tuple:
        """Load scene data from file path

        Args:
            file_path (str): Path to scene file.
            ins_max_num (int): Maximum number of instances per block.

        Returns:
            tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                A tuple of point data (N, 9), labels (N,), padded bounding box vertices (ins_max_num, 2, 3), and padded point masks (ins_max_num, N).
                The first three channels of point data are block normalized xyz, middle three are scene normalized xyz, and the last three are origianl xyz for visualization purpose.
                If the number of instances in a block is less than ins_max_num, the rest place in output vertices and mask will be filled as 0. That why we call 'padded'.
        """
        pc_xyz, ins_labels = DataUtils.load_sampled_file(file_path)
        block_num = pc_xyz.shape[0]

        ori_xyz = []
        bbvert_padded_labels = []
        pmask_padded_labels = []

        for block_id in range(block_num):
            # normalized and center xyz within the block
            pc_xyz[block_id] = DataUtils.normalize_xyz(pc_xyz[block_id])

            # reserved for final visualization
            ori_xyz.append(pc_xyz[block_id, :, 0:3].copy()) 
            
            # get bounding box vertices and point masks
            bbvert_padded_labels_single, pmask_padded_labels_single = DataLoader.get_bbvert_pmask_labels(
                pc_xyz[block_id],
                ins_labels[block_id],
                ins_max_num
            )

            bbvert_padded_labels.append(bbvert_padded_labels_single)
            pmask_padded_labels.append(pmask_padded_labels_single)

        ori_xyz = np.stack(ori_xyz, axis=0)
        pc_xyz = np.concatenate([pc_xyz, ori_xyz], axis=-1)
        bbvert_padded_labels = np.stack(bbvert_padded_labels, axis=0)
        pmask_padded_labels = np.stack(pmask_padded_labels, axis=0)

        return pc_xyz, ins_labels, bbvert_padded_labels, pmask_padded_labels

    @staticmethod
    def data_sample(data_sample_queue, input_list, ins_max_num, epoch, num_works, shuffle):
        """Get blocks in scenes and put into queue

        Args:
            data_sample_queue (multiprocessing.Manager.Queue): A Queue to put blocks.
            input_list (list): A list of file paths.
            ins_max_num (int): Maximum number of instances in one block.
            epoch (int): Number of epochs.
            num_works (int): Number of workers to load scene file simultaneously.
            shuffle (bool): Shuffle scenes every epoch or not.
        """
        input_list_length = len(input_list)
        num_work = min(num_works, multiprocessing.cpu_count())

        chunksize = input_list_length // num_work
        print("num input_list: {}, num works: {}, chunksize: {}".format(input_list_length, num_work, chunksize))

        def data_sample_single(input_file):
            datalabel = DataLoader.load_fixed_points(input_file, ins_max_num)
            return datalabel

        for ep in range(epoch):            
            if shuffle:
                np.random.seed(ep)
                np.random.shuffle(input_list)

            for idx in range(chunksize + 1):
                start_idx = min(idx * num_work, input_list_length)
                end_idx = min((idx + 1) * num_work, input_list_length)
                if start_idx >= input_list_length or end_idx > input_list_length:
                    continue

                with futures.ThreadPoolExecutor(num_work) as pool:
                    data_sem_ins = list(pool.map(data_sample_single, input_list[start_idx:end_idx], chunksize=1))

                    for dsi in data_sem_ins:
                        data_sample_queue.put(dsi)
                        del dsi
                        gc.collect()

                    pool.shutdown()
                    gc.collect()

    @staticmethod
    def data_prepare(data_sample_queue, data_queue, blocks, epoch, batch_size):
        """Prepare batches and put into queue

        Args:
            data_sample_queue (multiprocessing.Manager.Queue): A Queue to get blocks.
            data_queue (multiprocessing.Manager.Queue): A Queue to put batches.
            blocks (int): Number of all blocks.
            epoch (int): Number of epochs.
            batch_size (int): Batch size.
        """
        for _ in range(epoch):
            total_batch = DataLoader.get_total_batch_num(blocks, batch_size)

            pc_list = list()
            ins_labels_list = list()
            bbvert_padded_labels_list = list()
            pmask_padded_labels_list = list()

            while total_batch > 0:
                pc, ins_labels, bbvert_padded_labels, pmask_padded_labels = data_sample_queue.get()

                pc_list.append(pc)
                ins_labels_list.append(ins_labels)
                bbvert_padded_labels_list.append(bbvert_padded_labels)
                pmask_padded_labels_list.append(pmask_padded_labels)

                del pc
                del ins_labels
                del bbvert_padded_labels
                del pmask_padded_labels

                batch_pc = np.concatenate(pc_list, axis=0)
                batch_ins_labels = np.concatenate(ins_labels_list, axis=0)
                batch_bbvert_padded_labels = np.concatenate(bbvert_padded_labels_list, axis=0)
                batch_pmask_padded_labels = np.concatenate(pmask_padded_labels_list, axis=0)

                batch_data_length = batch_pc.shape[0]
                num_batch_size = batch_data_length // batch_size

                # put to data queue
                for idx in range(num_batch_size):
                    total_batch -= 1
                    start_idx = idx * batch_size
                    end_idx = (idx + 1) * batch_size
                    data_queue.put((
                        batch_pc[start_idx:end_idx, ...],
                        batch_ins_labels[start_idx:end_idx],
                        batch_bbvert_padded_labels[start_idx:end_idx, ...],
                        batch_pmask_padded_labels[start_idx:end_idx, ...]
                    ))

                # handle remaining blocks
                remainder = batch_data_length % batch_size
                if remainder:
                    pc_list = [batch_pc[-remainder:]]
                    ins_labels_list = [batch_ins_labels[-remainder:]]
                    bbvert_padded_labels_list = [batch_bbvert_padded_labels[-remainder:]]
                    pmask_padded_labels_list = [batch_pmask_padded_labels[-remainder:]]
                else:
                    pc_list = list()
                    ins_labels_list = list()
                    bbvert_padded_labels_list = list()
                    pmask_padded_labels_list = list()
                
                # directly put the last batch into queue
                # so the batch size of the last batch may be different to previous batches
                # it should be considered when training with multiple GPUs
                if total_batch == 1:
                    total_batch = 0
                    start_idx = num_batch_size * batch_size
                    end_idx = (num_batch_size + 1) * batch_size
                    data_queue.put((
                        batch_pc[start_idx:end_idx, ...],
                        batch_ins_labels[start_idx:end_idx],
                        batch_bbvert_padded_labels[start_idx:end_idx, ...],
                        batch_pmask_padded_labels[start_idx:end_idx, ...]
                    ))

                del batch_pc
                del batch_ins_labels
                del batch_bbvert_padded_labels
                del batch_pmask_padded_labels

                gc.collect()

    def get_batch(self) -> tuple:
        """Get a batch

        Raises:
            Exception: This function should be called after init_data_pipeline().

        Returns:
            tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
                A tuple of batch point data (B, N, 9),
                batch labels (B, N),
                batch padded bounding box vertices (B, ins_max_num, 2, 3),
                and batch padded point masks (B, ins_max_num, N).
        """
        if not hasattr(self, 'producer_process') or not hasattr(self, 'consumer_process'):
            raise Exception('Please call init_data_pipeline() first!')
        
        bat_pc, bat_ins_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels = self.data_queue.get()
        return bat_pc, bat_ins_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels


class DataUtils:
    """Useful functions for data processing
    """
    ###########################################
    ############## Data Sampling ##############
    ###########################################

    @staticmethod
    def clip_data(data: numpy.ndarray, xmin=-sys.float_info.max, ymin=-sys.float_info.max, xmax=sys.float_info.max, ymax=sys.float_info.max) -> numpy.ndarray:
        """Clip data. Only keep the data in the boundary

        Args:
            data (np.ndarray): Data (N, 2). The number of channels can be larger than 2, but the first three must be x and y.
            xmin (float, optional): Minimum x value. Defaults to -sys.float_info.max.
            ymin (float, optional): Minimum y value. Defaults to -sys.float_info.max.
            xmax (float, optional): Maximum x value. Defaults to sys.float_info.max.
            ymax (float, optional): Maximum y value. Defaults to sys.float_info.max.

        Returns:
            np.ndarray: Clipped data with the same shape as input.
        """
        if len(data) > 0:
            cond = (
                (data[:, 0] >= xmin) &
                (data[:, 1] >= ymin) &
                (data[:, 0] <= xmax) &
                (data[:, 1] <= ymax)
            )
            return data[cond]
        else:
            return data

    @staticmethod
    def random_sample(data: numpy.ndarray, num_samples: int) -> numpy.ndarray:
        """Randomly sample data.

        Args:
            data (numpy.ndarray): Data (N, C).
            num_samples (int): Number of sample size.

        Returns:
            numpy.ndarray: Sampled data (num_samples, C).
        """
        n = data.shape[0]
        if n >= num_samples:
            indices = np.random.choice(n, num_samples, replace=False)
        # if number of data is less than num_samples, duplicate random data to get enough output
        else:
            indices = np.random.choice(n, num_samples - n, replace=True)
            indices = list(range(n)) + list(indices)
            np.random.shuffle(indices)
        sampled = data[indices, :]

        return sampled

    @staticmethod
    def random_sample_with_density_constaint(data: numpy.ndarray, num_samples: int, search_radius=0.05, low_density_points_ratio=0.25) -> np.ndarray:
        """Sample data with Inverse Density Importance Sampling and random sampling

        Args:
            data (numpy.ndarray): Input data to be sampled (N, C).
            num_samples (int): Number of sample size.
            search_radius (float, optional): Search radius to find neighbor points for density estimation. Defaults to 0.05.
            low_density_points_ratio (float, optional): After density estimation, data will be sorted from low to high density, 
                and the first N' points will be taken for output, where N' = low_density_points_ratio * N.
                Then, the rest of (num_samples - N') output points will be randomly sampled.
                Defaults to 0.25.

        Returns:
            np.ndarray: Sampled data. (num_samples, C).
        """
        # density based on neighbor number in certain radius
        neigh = NearestNeighbors(radius=search_radius)
        neigh.fit(data[:,0:3])

        rng = neigh.radius_neighbors(data[:,0:3], return_distance=False)
        rng_counts = [len(n) for n in rng]
        rng_counts = np.asarray(rng_counts)

        order = np.argsort(rng_counts)

        # point numbers of low/high density points
        num_low_density_pt = int(num_samples * low_density_points_ratio)
        num_random_sample_pt = num_samples - num_low_density_pt

        # subsample
        num_data = data.shape[0]
        if num_data <= num_low_density_pt:
            print('points are not enough, use all points')
            sampled_data = DataUtils.random_sample(data, num_samples=num_samples)
        else:
            low_density_pt_ind = order[:num_low_density_pt]
            high_density_pt_ind = order[num_low_density_pt:]
            
            low_density_data = data[low_density_pt_ind, :]
            sampled_high_density_data = DataUtils.random_sample(data[high_density_pt_ind, :], num_samples=num_random_sample_pt)

            sampled_data = np.concatenate([low_density_data, sampled_high_density_data], axis=0)
        
        return sampled_data

    @staticmethod
    def filter_instance(data: numpy.ndarray, filter_bbox_size=[0, 0, 0], filter_ins_pt_counts=50) -> numpy.ndarray:
        """Filter instance data

        Args:
            data (numpy.ndarray): Input data. (N, 4), where the channels are x, y, z, and intance labels.
            filter_bbox_size (list, optional): A list of box size (width, height, depth) should be filtered out. Defaults to [0, 0, 0].
            filter_ins_pt_counts (int, optional): Instances whose number of points are less than filter_ins_pt_counts will be filtered out. Defaults to 50.

        Returns:
            numpy.ndarray: Filtered data. (N', 4).
        """
        if len(data.shape) <= 3:
            return data

        all_labels = data[:,-1].reshape(-1)
        ins_labels, pt_ins_ind, ins_pt_counts = np.unique(all_labels, return_inverse=True, return_counts=True)
        ins_pt_ind = np.split(np.argsort(pt_ins_ind), np.cumsum(ins_pt_counts[:-1]))

        kept_indices = []
        kept_labels = []
        for label, pt_counts, pt_ind in zip(ins_labels, ins_pt_counts, ins_pt_ind):
            ins_pt_xyz = data[pt_ind, 0:3]
            bbox_size = np.max(ins_pt_xyz, axis=0) - np.min(ins_pt_xyz, axis=0)
            
            if filter_bbox_size is not None and np.any(np.equal(bbox_size, filter_bbox_size)):
                continue
            if filter_ins_pt_counts is not None and pt_counts < filter_ins_pt_counts:
                continue

            kept_indices.extend(pt_ind)
            kept_labels.append(label)
        
        return data[kept_indices, :]

    @staticmethod
    def sample_data(data, num_samples, min_instance_points_num=50, use_idis=False):
        # filter out instances with few points or invalid bbox
        filtered_data = DataUtils.filter_instance(data, filter_bbox_size=[0,0,0], filter_ins_pt_counts=min_instance_points_num)

        # sub-sample cloud
        if use_idis:
            sampled_data = DataUtils.random_sample_with_density_constaint(filtered_data, num_samples)
        else:
            sampled_data = DataUtils.random_sample(filtered_data, num_samples)

        # filter out instances with few points or invalid bbox
        sampled_data = DataUtils.filter_instance(sampled_data, filter_bbox_size=[0,0,0], filter_ins_pt_counts=min_instance_points_num)

        # get enough points and random shuffle
        sampled_data = DataUtils.random_sample(sampled_data, num_samples)

        return sampled_data


    ###########################################
    ############### Data Loader ###############
    ###########################################
    
    @staticmethod
    def load_xyz(scene_filepath: str) -> np.ndarray:
        """Load xyz file

        Args:
            scene_filepath (str): Path to .xyz file

        Returns:
            np.ndarray: Loaded data.
        """
        return np.loadtxt(scene_filepath)  # x, y, z, object id

    @staticmethod
    def load_h5(scene_filepath: str) -> np.ndarray:
        """Load HDF5 file

        Args:
            scene_filepath (str): Path to HDF5 file.
                The file must have at least one key, which contains data with x, y, z, and instance labels (N, 4);
                Or two keys: the first key of the file is xyz data (N, 3), and the second key is instance labels (N,).

        Returns:
            np.ndarray: Loaded data. (N, 4).
        """
        fin = h5py.File(scene_filepath, 'r')
        keys = list(fin.keys())

        if len(keys) == 0:
            return np.array([])
        elif len(keys) == 1:
            return fin[keys[0]][:]
        else:
            xyz = fin[keys[0]][:]
            labels = fin[keys[1]][:].reshape([-1, 1])
            return np.concatenate((xyz, labels), axis=1)

    @staticmethod
    def load_npy(scene_filepath: str) -> np.ndarray:
        """Load npy file

        Args:
            scene_filepath (str): Path to .npy file.

        Returns:
            np.ndarray: Loaded data.
        """
        return np.load(scene_filepath)

    @staticmethod
    def load_scene_file(scene_filepath: str,
                        data_format=None,
                        xmin=-sys.float_info.max,
                        ymin=-sys.float_info.max,
                        xmax=sys.float_info.max,
                        ymax=sys.float_info.max) -> np.ndarray:
        """Load scene file

        Args:
            scene_filepath (str): Path to scene file.
            data_format (str, optional): 'xyz', 'h5', or 'npy'. If None, get extension automatically. 
                If not support data format, return an empty array. Defaults to None.
            xmin (float, optional): Minimum x value. Defaults to -sys.float_info.max.
            ymin (float, optional): Minimum y value. Defaults to -sys.float_info.max.
            xmax (float, optional): Maximum x value. Defaults to sys.float_info.max.
            ymax (float, optional): Maximum y value. Defaults to sys.float_info.max.

        Returns:
            np.ndarray: Loaded data.
        """
        
        if data_format is None:
            data_format = os.path.splitext(scene_filepath)[-1][1:]
            
        if data_format == 'xyz':
            scene = DataUtils.load_xyz(scene_filepath)

        elif data_format == 'h5':
            scene = DataUtils.load_h5(scene_filepath)

        elif data_format == 'npy':
            scene = DataUtils.load_npy(scene_filepath)

        else:
            print('{}: format of input filename is not consistent with the data_format'.format(scene_filepath))
            scene = np.array([])

        return DataUtils.clip_data(scene, xmin, ymin, xmax, ymax)

    @staticmethod
    def load_scene_dir(scene_dir: str,
                       data_format='xyz',
                       xmin=-sys.float_info.max,
                       ymin=-sys.float_info.max,
                       xmax=sys.float_info.max,
                       ymax=sys.float_info.max) -> np.ndarray:
        """Load scene directory

        Args:
            scene_dir (str): Path to scene directory. The files in the same directory will be merged as one scene data.
            data_format (str, optional): 'xyz', 'h5', or 'npy'. If None, get extension automatically. 
                If not support data format, return an empty array. Defaults to 'xyz'.
            xmin (float, optional): Minimum x value. Defaults to -sys.float_info.max.
            ymin (float, optional): Minimum y value. Defaults to -sys.float_info.max.
            xmax (float, optional): Maximum x value. Defaults to sys.float_info.max.
            ymax (float, optional): Maximum y value. Defaults to sys.float_info.max.

        Returns:
            np.ndarray: Loaded data.
        """
        # get filenames
        all_sub_scene_filenames = glob.glob(os.path.join(scene_dir, '*.' + data_format))
        if len(all_sub_scene_filenames) == 0:
            return np.array([], dtype=np.float32)

        # load each file
        all_sub_scene_data = []
        for sub_scene_filename in all_sub_scene_filenames:
            sub_scene = DataUtils.load_scene_file(sub_scene_filename, data_format, xmin, ymin, xmax, ymax)
            all_sub_scene_data.append(sub_scene)

        return np.concatenate(all_sub_scene_data, axis=0)

    @staticmethod
    def load_sampled_file(filepath: str) -> np.ndarray:
        """Load sampled file

        Args:
            filepath (str): Path to sampled file. The file must have three keys:
                the first key is 'coords' which stores xyz data of blocks (B, N, 3).
                the second key is 'points' which stores block centered and scene normalized xyz data of blocks (B, N, 6).
                the third key is 'labels' which stores instance labels of blocks (B, N).

        Returns:
            np.ndarray: Loaded sampled data (B, N, 6). The channels are x, y, z, scene normalized x, scene normalized y, scene normalized z.
        """

        fin = h5py.File(filepath, 'r')
        coords = fin['coords'][:]
        points = fin['points'][:]
        ins_labels = fin['labels'][:]
        fin.close()
        
        data = np.concatenate([coords, points[:, :, 3:6]], axis=-1)  # global coords and normalized coords in scenes
        
        return data, ins_labels

    @staticmethod
    def load_prediction_file(filepath: str) -> tuple:
        """Load prediction file from Tester

        Args:
            filepath (str): Path to prediction file (*.h5). The file should be generated from Tester.

        Returns:
            tuple (list, list, list):
                A tuple of list of scene normalized xyz (N, 3),
                list of ground truth instance labels (N),
                and list of predicted instance labels (B, N), 
                where B is number of blocks and N is number of points in each block. 
                Notice that the N in each block which may be different if predicted by Iterative Prediction mechanism.
        """
        fin = h5py.File(filepath, 'r')
        pts = []
        ins_gt = []
        ins_pred = []

        for key in list(fin.keys()):
            data = fin[key][:]
            assert data.shape[-1] == 5, 'Failed to load prediction {}, the shape should be (N, 5)'.format(key)

            pts.append(data[:,0:3])
            ins_gt.append(data[:,3].astype(np.int32))
            ins_pred.append(data[:,4].astype(np.int32))
        
        fin.close()

        return pts, ins_gt, ins_pred


    ##########################################
    ################ Others ##################
    ##########################################

    @staticmethod
    def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
        """Normalize data

        Args:
            xyz (np.ndarray): Input data. (N, 3). The channels can be larger than 3, but the first three must be x, y, z.

        Returns:
            np.ndarray: Normalized data whose xyz are normalized into [0, 1].
        """
        xyz_normalized = xyz.copy()

        min_xyz = np.min(xyz[:,0:3], axis=0, keepdims=True)
        max_xyz = np.max(xyz[:,0:3], axis=0, keepdims=True)
        whd = np.maximum((max_xyz - min_xyz), 1e-3)

        xyz_normalized[:,0:3] = (xyz_normalized[:,0:3] - min_xyz) / whd

        return xyz_normalized
