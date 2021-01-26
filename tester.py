import os

import h5py
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from BoNet import BoNet
from helper_data import DataLoader, DataConfigs


class SimpleTester:
    """A generic tester
    """
    def __init__(self,
                 model_path: str,
                 data_configs: DataConfigs,
                 batch_size=2):
        """Create Tester

        Args:
            model_path (str): Path to model weight (*.ckpt).
            data_configs (DataConfigs): Data configuration.
            batch_size (int, optional): Batch size when inference. Defaults to 2.
        """
        self.net = self.load_net(model_path, data_configs)
        self.batch_size = batch_size

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' # specify the GPU to use

    def load_net(self, model_path: str, data_configs: DataConfigs) -> BoNet:
        """Load 3D-BoNet

        Args:
            model_path (str): Path to model weight (*.ckpt).
            data_configs (DataConfigs): Data configuration.

        Raises:
            Exception: If model_path doesn't exist.

        Returns:
            BoNet: 3D-BoNet for inference.
        """
        # check model exists or not
        if not os.path.isfile(model_path + ".data-00000-of-00001"):
            print("There is no model at {}".format(model_path))
            raise Exception("model path error")

        # Create 3D-BoNet
        tf.reset_default_graph()
        net = BoNet(configs=data_configs)

        # Define network architecture
        net.X_pc = tf.placeholder(shape=[None, None, net.points_cc], dtype=tf.float32, name="X_pc")
        
        with tf.variable_scope('backbone'):
            net.point_features, net.global_features = net.backbone_pointnet2(net.X_pc)
        
        with tf.variable_scope('bbox'):
            net.y_bbvert_pred_raw, net.y_bbscore_pred_raw = net.bbox_net(net.global_features)
        
        with tf.variable_scope('pmask'):
            net.y_pmask_pred_raw = net.pmask_net(net.point_features, net.global_features, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw)

        # Restore trained model
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = '0'
        config.gpu_options.allow_growth = True
        net.sess = tf.Session(config=config)

        tf.train.Saver().restore(net.sess, model_path)
        print('Model restored sucessful!')

        return net

    def inference(self, batch_data: np.ndarray) -> tuple:
        """Inference a batch data

        Args:
            batch_data (np.ndarray): Batch data. (B, N, 6).

        Returns:
            tuple (np.ndarray, np.ndarray, np.ndarray): Predictions of bounding box vertices, bounding box scores, and point masks.
        """
        assert len(batch_data.shape) == 3

        [batch_bbvert_raw, batch_bbscore_raw, batch_pmask_raw] = \
            self.net.sess.run([self.net.y_bbvert_pred_raw, self.net.y_bbscore_pred_raw, self.net.y_pmask_pred_raw], \
                feed_dict={self.net.X_pc: batch_data[:, :, 0:6]})

        return batch_bbvert_raw, batch_bbscore_raw, batch_pmask_raw
    
    def get_ins_pred(self, batch_data: np.ndarray) -> np.ndarray:
        """Get instance labels of batch data

        Args:
            batch_data (np.ndarray): A batch data. (B, N, 6).

        Returns:
            np.ndarray: Predicted instance labels of batch data. (B, N).
        """
        _, batch_bbscore, batch_pmask = self.inference(batch_data)

        batch_ins_pred = []
        for bbscore, pmask in zip(batch_bbscore, batch_pmask):
            pmask_new = pmask * np.tile(bbscore[:, None], [1, pmask.shape[-1]])
            ins_pred = np.argmax(pmask_new, axis=-2).astype(np.int32)
            
            batch_ins_pred.append(ins_pred)
        
        return np.stack(batch_ins_pred)

    def save_data(self, data_name: str, fout: h5py.File, pts: np.ndarray, ins_gt: np.ndarray, ins_pred: np.ndarray):
        """Save predictions to HDF5 file

        Args:
            data_name (str): Data name in HDF5 file.
            fout (h5py.File): An instance of h5py.File in writable mode.
            pts (np.ndarray): Point data to be stored. (N, C).
            ins_gt (np.ndarray): Ground truth instance labels to be stored. (N,).
            ins_pred (np.ndarray): Predicted instance labels to be stored. (N,).
        """
        if fout is None: return
        
        ins_gt = np.reshape(ins_gt, [-1, 1])
        ins_pred = np.reshape(ins_pred, [-1, 1])
        output_data = np.concatenate([pts, ins_gt, ins_pred], axis=-1)
        
        fout.create_dataset(data_name, data=output_data, compression='gzip', dtype='float32', compression_opts=9)

    def inference_data(self, data_loader: DataLoader, fout=None) -> tuple:
        """Inference data from DataLoader

        Args:
            data_loader (DataLoader): An instance of DataLoader.
            fout (h5py.File, optional): An instance of h5py.File in writable mode. Defaults to None.

        Returns:
            tuple (list, list, list): A tuple of list of data from data_loader, list of ground truth instance labels, and list of predicted instance labels
        """
        block_id = 0
        data_all = []
        ins_gt_all = []
        ins_pred_all = []

        data_loader.init_data_pipeline()

        for _ in tqdm(range(data_loader.total_batch_num)):
            batch_data, batch_ins_labels, _, _ = data_loader.get_batch()
            batch_ins_pred = self.get_ins_pred(batch_data)

            # store data
            for data, ins_labels, ins_pred in zip(batch_data, batch_ins_labels, batch_ins_pred):
                data_all.append(data[:,3:6])
                ins_gt_all.append(ins_labels)
                ins_pred_all.append(ins_pred)

                data_name = 'block_' + str(block_id).zfill(4) + '_00'
                self.save_data(data_name, fout, data[:,3:6], ins_labels, ins_pred)

                block_id += 1

        return data_all, ins_gt_all, ins_pred_all
    
    def test(self, file_path: str, save_predictions=False, output_basename=None) -> tuple:
        """Test a file

        Args:
            file_path (str): Path to file (*.h5).
            save_predictions (bool, optional): Save predictions to storage or not. Defaults to False.
            output_basename (str, optional): Output basename of prediction file. Ignored if save_predictions = False. Defaults to None.

        Returns:
            tuple (list, list, list): A tuple of list of data from data_loader, list of ground truth instance labels, and list of predicted instance labels
        """
        data = DataLoader(filepaths=[file_path], epoch=1, batch_size=self.batch_size, num_works=1, shuffle=False)

        # Create output file if needed
        if output_basename is None:
            output_basename = os.path.splitext(file_path)[0]

        if save_predictions:
            output_filename = output_basename + '_blocks_pred.h5'
            fout = h5py.File(output_filename, 'w')
            print('Save block predictions to', output_filename)
        else:
            fout = None

        # Start processing
        data_all, ins_gt_all, ins_pred_all = self.inference_data(data, fout)

        # Close output file
        if fout is not None: fout.close()

        return data_all, ins_gt_all, ins_pred_all


class IterativeTester(SimpleTester):
    """A Tester using Iterative Prediction mechanism
    """
    def __init__(self,
                 model_path: str,
                 data_configs: DataConfigs,
                 min_prob=0.8,
                 min_instance_pts_num=50,
                 max_iteration=10):
        """Create IterativeTester

        Args:
            model_path (str): Path to model weight (*.ckpt).
            data_configs (DataConfigs): Data configuration.
            min_prob (float, optional): Minimum probability score for Iterative Prediction. Defaults to 0.8.
            min_instance_pts_num (int, optional): Minimum point number of instance for Iterative Prediction. Defaults to 50.
            max_iteration (int, optional): Maximum iteration for Iterative Prediction. Defaults to 10.
        """
        super().__init__(model_path, data_configs, 1) # batch size can only be 1
        
        self.min_prob = min_prob
        self.min_instance_pts_num = min_instance_pts_num
        self.max_iteration = max_iteration
    
    def get_ins_pred(self, data: np.ndarray, ins_labels: np.ndarray, verbose=False) -> list:
        """Get list of predictions by Iterative Prediction

        Args:
            data (np.ndarray): One block data. (N, 6). Notice: there is no 'Batch' dimension.
            ins_labels (np.ndarray): Instance labels of data. (N,).
            verbose (bool, optional): Display processing info or not. Defaults to False.

        Returns:
            list (tuple(np.ndarray, np.ndarray, np.ndarray)): A list of tuples, which contains
                new block data (N', 6), new block ground truth labels (N'), predicted labels of new block (N').
                Notice that the N' in different tuple may be different due to Iterative Prediction mechanism.
        """
        input_data = np.copy(data)
        input_labels = np.copy(ins_labels)

        remain_pts_num = input_data.shape[0]
        outputs = [] # data, gt_labels, pred_labels

        #volume = create_volume(cell_size)
        for iter in range(self.max_iteration):
            if verbose: print('iter {}, predict {} pts'.format(iter, input_data.shape[0]))

            # inference
            _, bbscore_pred_raw, pmask_pred_raw = self.inference(np.expand_dims(input_data, axis=0))
            bbscore_pred_raw = np.squeeze(bbscore_pred_raw, axis=0)
            pmask_pred_raw = np.squeeze(pmask_pred_raw, axis=0)

            # instance prediction
            pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
            ins_pred_tmp = np.argmax(pmask_pred, axis=-2).astype(np.int32)

            # mark labels of negative points as -1
            ins_pred_p_cond = pmask_pred.max(axis=-2) >= self.min_prob
            ins_pred_tmp[~ins_pred_p_cond] = -1

            if verbose:
                print('positive instance pts:', np.sum(ins_pred_p_cond))
                print('negative instance pts:', np.sum(~ins_pred_p_cond))
            if np.sum(ins_pred_p_cond) == 0: break

            # check points number of positive instance
            ins_labels, pt_ins_ind, ins_pt_count = np.unique(ins_pred_tmp, return_inverse=True, return_counts=True, axis=0)
            ins_pt_ind = np.split(np.argsort(pt_ins_ind), np.cumsum(ins_pt_count[:-1]))

            for label, pt_count, pt_ind in zip(ins_labels, ins_pt_count, ins_pt_ind):
                if label == -1 or pt_count < self.min_instance_pts_num:
                    ins_pred_p_cond[pt_ind] = False

            ins_pred_p = ins_pred_tmp[ins_pred_p_cond]
            ins_pred_p_ind = np.argwhere(ins_pred_p_cond).reshape(-1)
            ins_pred_n_ind = np.argwhere(~ins_pred_p_cond).reshape(-1)
            if verbose:
                print('positive instance pts (after filtering):', len(ins_pred_p_ind))
                print('negative instance pts (after filtering):', len(ins_pred_n_ind))
            if len(ins_pred_p) == 0: break

            # store data for saving output
            outputs.append((input_data[ins_pred_p_cond, :], input_labels[ins_pred_p_cond], ins_pred_p))
            
            # update data
            input_data = input_data[~ins_pred_p_cond, :]
            input_labels = input_labels[~ins_pred_p_cond]

            # check loop-breaking conditions
            # 1. previous remain point number is close to current negative point number
            if remain_pts_num - len(ins_pred_n_ind) < self.min_instance_pts_num:
                break
            # 2. there is no negative instance
            if len(ins_pred_n_ind) == 0:
                break

            # update variables
            remain_pts_num = len(ins_pred_n_ind)
            #remain_pts_indices = remain_pts_indices[ins_pred_n_ind]            

        # get instance label of all block points
        #xyz_block_int = (original_input_data[:, 3:6] / cell_size).astype(np.int32)
        #ins_pred = volume[tuple(xyz_block_int.T)]

        outputs.append((input_data, input_labels, -1 * np.ones_like(input_labels, dtype=np.int32)))

        return outputs


    def inference_data(self, data_loader: DataLoader, fout=None):
        """Inference data from DataLoader

        Args:
            data_loader (DataLoader): An instance of DataLoader.
            fout (h5py.File, optional): An instance of h5py.File in writable mode. Defaults to None.

        Returns:
            tuple (list, list, list): A tuple of list of data from data_loader, list of ground truth instance labels, and list of predicted instance labels
        """
        data_all = []
        ins_gt_all = []
        ins_pred_all = []

        data_loader.batch_size = 1 # force batch size to 1 because Iterative Prediction can only handle 1 batch once time
        data_loader.init_data_pipeline()

        for block_id in tqdm(range(data_loader.total_batch_num)):
            batch_data, batch_ins_labels, _, _ = data_loader.get_batch()
            results = self.get_ins_pred(batch_data[0], batch_ins_labels[0])

            # store data
            for i, (data, ins_gt, ins_pred) in enumerate(results):
                data_all.append(data[:,3:6])
                ins_gt_all.append(ins_gt)
                ins_pred_all.append(ins_pred)

                data_name = 'block_' + str(block_id).zfill(4) + '_' + str(i).zfill(2)
                self.save_data(data_name, fout, data[:,3:6], ins_gt, ins_pred)

        return data_all, ins_gt_all, ins_pred_all
