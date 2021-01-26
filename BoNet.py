
import os
import glob
import shutil
from typing import Tuple

import tensorflow as tf

from helper_net import Ops as Ops
from helper_data import DataConfigs


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py

    Args:
        tower_grads (List of lists of (gradient, variable) tuples): The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.

    Returns:
        List of (gradient, variable) tuples: the gradient which has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        # for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class BoNet:
    """3D-BoNet
    """
    def __init__(self, configs: DataConfigs):
        """Initialize 3D-BoNet with dataset config

        Args:
            configs (DataConfigs): Dataset Config.
        """
        self.points_cc = configs.points_cc # point channels
        self.bb_num = configs.ins_max_num # max instance numbers

    def creat_folders(self, name='log', re_train=False):
        """Create folders for storing models and summaries

        Args:
            name (str, optional): Folder name. Defaults to 'log'.
            re_train (bool, optional): If True, the files in the existed folder will be kept. Defaults to False.
        """
        self.train_mod_dir = os.path.join('.', name, 'train_mod')
        self.train_sum_dir = os.path.join('.', name, 'train_sum')
        self.test_sum_dir = os.path.join('.', name, 'test_sum')
        self.re_train = re_train
        
        print("re_train:", self.re_train)

        def check_dirs(path):
            if os.path.exists(path):
                if self.re_train:
                    print(path, ": files kept!")
                else:
                    shutil.rmtree(path)
                    os.makedirs(path)
                    print(path, ': deleted and then created!')
            else:
                os.makedirs(path)
                print(path, ': created!')
        
        check_dirs(self.test_sum_dir)
        check_dirs(self.train_sum_dir)
        check_dirs(self.train_mod_dir)

    # 1. backbone
    def backbone_pointnet2(self, X_pc) -> tuple:
        """PointNet++ backbone

        Args:
            X_pc (tf.Tensor): The input point cloud (B, N, C)

        Returns:
            Tuple: Tuple of (point features, global features). (B, N, 128), (B, 512)
        """
        import helper_pointnet2 as pnet2
        l0_xyz = X_pc[:, :, 0:3]
        l0_points = X_pc[:, :, 3:self.points_cc]

        # Set Abstraction (SA) layers
        l1_xyz, l1_points, _ = pnet2.pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.1, nsample=64, mlp=[32, 32, 64], mlp2=None, group_all=False, is_training=None, bn_decay=None, scope='layer1')
        l2_xyz, l2_points, _ = pnet2.pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.2, nsample=128, mlp=[64, 64, 128], mlp2=None, group_all=False, is_training=None, bn_decay=None, scope='layer2')
        l3_xyz, l3_points, _ = pnet2.pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4, nsample=256, mlp=[128, 128, 256], mlp2=None, group_all=False, is_training=None, bn_decay=None, scope='layer3')
        l4_xyz, l4_points, _ = pnet2.pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[256, 256, 512], mlp2=None, group_all=True, is_training=None, bn_decay=None, scope='layer4')

        # Feature Propagation (FP) layers
        l3_points = pnet2.pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], is_training=None, bn_decay=None, scope='fa_layer1')
        l2_points = pnet2.pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], is_training=None, bn_decay=None, scope='fa_layer2')
        l1_points = pnet2.pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training=None, bn_decay=None, scope='fa_layer3')
        l0_points = pnet2.pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz, l0_points], axis=-1), l1_points, [128, 128, 128, 128], is_training=None, bn_decay=None, scope='fa_layer4')
                
        global_features = tf.reshape(l4_points, [-1, 512])
        point_features = l0_points

        return point_features, global_features
        
    # 2. bbox
    def bbox_net(self, global_features: tf.Tensor) -> tuple:
        """Bounding box branch

        Args:
            global_features (tf.Tensor): Global features from backbone

        Returns:
            tuple: Tuple of vertices and scores of bounding boxes. (B, bb_num, 2, 3), (B, bb_num)
        """
        b1 = Ops.xxlu(Ops.fc(global_features, out_d=512, name='b1'), label='lrelu')
        b2 = Ops.xxlu(Ops.fc(b1, out_d=256, name='b2'), label='lrelu')

        # sub branch 1
        b3 = Ops.xxlu(Ops.fc(b2, out_d=256, name='b3'), label='lrelu')
        bbvert = Ops.fc(b3, out_d=self.bb_num * 2 * 3, name='bbvert')
        bbvert = tf.reshape(bbvert, [-1, self.bb_num, 2, 3])
        points_min = tf.reduce_min(bbvert, axis=-2)[:, :, None, :]
        points_max = tf.reduce_max(bbvert, axis=-2)[:, :, None, :]
        y_bbvert_pred = tf.concat([points_min, points_max], axis=-2, name='y_bbvert_pred')

        # sub branch 2
        b4 = Ops.xxlu(Ops.fc(b2, out_d=256, name='b4'), label='lrelu')
        y_bbscore_pred = tf.sigmoid(Ops.fc(b4, out_d=self.bb_num * 1, name='y_bbscore_pred'))

        return y_bbvert_pred, y_bbscore_pred

    # 3. pmask
    def pmask_net(self, point_features: tf.Tensor, global_features: tf.Tensor, bbox: tf.Tensor, bboxscore: tf.Tensor) -> tf.Tensor:
        """Point mask branch

        Args:
            point_features (tf.Tensor): Point features from backbone. (B, N, C1)
            global_features (tf.Tensor): Global features from backbone. (B, C2)
            bbox (tf.Tensor): Bounding box predictions from bbox_net. (B, bb_num, 2, 3)
            bboxscore (tf.Tensor): Bounding box scores predictions from bbox_net. (B, bb_num)

        Returns:
            tf.Tensor: Point mask predictions. (B, bb_num, N)
        """
        p_f_num = int(point_features.shape[-1])
        p_num = tf.shape(point_features)[1]
        bb_num = int(bbox.shape[1])

        global_features = tf.tile(Ops.xxlu(Ops.fc(global_features, out_d=256, name='down_g1'), label='lrelu')[:, None, None, :], [1, p_num, 1, 1])
        point_features = Ops.xxlu(Ops.conv2d(point_features[:, :, :, None], k=(1, p_f_num), out_c=256, str=1, name='down_p1', pad='VALID'), label='lrelu')
        point_features = tf.concat([point_features, global_features], axis=-1)
        point_features = Ops.xxlu(Ops.conv2d(point_features, k=(1, int(point_features.shape[-2])), out_c=128, str=1, pad='VALID', name='down_p2'), label='lrelu')
        point_features = Ops.xxlu(Ops.conv2d(point_features, k=(1, int(point_features.shape[-2])), out_c=128, str=1, pad='VALID', name='down_p3'), label='lrelu')
        point_features = tf.squeeze(point_features, axis=-2)

        bbox_info = tf.tile(tf.concat([tf.reshape(bbox, [-1, bb_num, 6]), bboxscore[:, :, None]], axis=-1)[:, :, None, :], [1, 1, p_num, 1])
        pmask0 = tf.tile(point_features[:, None, :, :], [1, bb_num, 1, 1])
        pmask0 = tf.concat([pmask0, bbox_info], axis=-1)
        pmask0 = tf.reshape(pmask0, [-1, p_num, int(pmask0.shape[-1]), 1])

        pmask1 = Ops.xxlu(Ops.conv2d(pmask0, k=(1, int(pmask0.shape[-2])), out_c=64, str=1, pad='VALID', name='pmask1'), label='lrelu')
        pmask2 = Ops.xxlu(Ops.conv2d(pmask1, k=(1, 1), out_c=32, str=1, pad='VALID', name='pmask2'), label='lrelu')
        pmask3 = Ops.conv2d(pmask2, k=(1, 1), out_c=1, str=1, pad='VALID', name='pmask3')
        pmask3 = tf.reshape(pmask3, [-1, bb_num, p_num])

        y_pmask_logits = pmask3
        y_pmask_pred = tf.nn.sigmoid(y_pmask_logits, name='y_pmask_pred')

        return y_pmask_pred

    # get complete model
    def get_model(self, X_pc: tf.Tensor, Y_bbvert: tf.Tensor) -> tuple:
        """Get complete model

        Args:
            X_pc (tf.Tensor): The input point cloud. (B, N, C)
            Y_bbvert (tf.Tensor): Ground truth of bounding box. (B, bb_num, 2, 3)

        Returns:
            tuple: Predictions of (bbox, bbox score, ordered bbox, bbox order, orederd bbox score, ordered point mask, point mask)
        """
        # 2. define networks, losses
        with tf.variable_scope('backbone'):
            point_features, global_features = self.backbone_pointnet2(X_pc)

        with tf.variable_scope('bbox'):
            y_bbvert_pred_raw, y_bbscore_pred_raw = self.bbox_net(global_features)

            # association, only used for training
            bbox_criteria = 'use_all_ce_l2_iou'
            y_bbvert_pred, pred_bborder = Ops.bbvert_association(X_pc,  y_bbvert_pred_raw, Y_bbvert, label=bbox_criteria)
            y_bbscore_pred = Ops.bbscore_association(y_bbscore_pred_raw, pred_bborder)

        with tf.variable_scope('pmask'):
            y_pmask_pred = self.pmask_net(point_features, global_features, y_bbvert_pred, y_bbscore_pred)

        with tf.variable_scope('pmask', reuse=True):
            # during testing, no need to associate, use unordered predictions
            y_pmask_pred_raw = self.pmask_net(point_features, global_features, y_bbvert_pred_raw, y_bbscore_pred_raw)

        return y_bbvert_pred_raw, y_bbscore_pred_raw, y_bbvert_pred, pred_bborder, y_bbscore_pred, y_pmask_pred, y_pmask_pred_raw

    # get losses
    def get_loss(self, X_pc: tf.Tensor, Y_bbvert: tf.Tensor, Y_pmask: tf.Tensor, y_bbvert_pred: tf.Tensor, y_bbscore_pred: tf.Tensor, y_pmask_pred: tf.Tensor) -> tuple:
        """Get all losses

        Args:
            X_pc (tf.Tensor): The input point cloud. (B, N, C)
            Y_bbvert (tf.Tensor): Ground truth of bounding box. (B, bb_num, 2, 3)
            Y_pmask (tf.Tensor): Ground truth of point mask. (B, bb_num, N)
            y_bbvert_pred (tf.Tensor): Prediction of bounding box. (B, bb_num, 2, 3)
            y_bbscore_pred (tf.Tensor): Prediction of bounding box score. (B, bb_num)
            y_pmask_pred (tf.Tensor): Prediction of point mask. (B, bb_num, N)

        Returns:
            tuple: Losses of (bbox_all, bbox_l2, bbox_ce, bbox_iou, bbox score, point_mask), where bbox_all = bbox_l2 + bbox_ce + bbox_iou
        """
        with tf.variable_scope('bbox'):
            bbox_criteria = 'use_all_ce_l2_iou'
            bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou = \
                Ops.get_loss_bbvert(X_pc, y_bbvert_pred, Y_bbvert, label=bbox_criteria)
            bbscore_loss = Ops.get_loss_bbscore(y_bbscore_pred, Y_bbvert)

            # summary
            tf.summary.scalar('bbvert_loss', bbvert_loss)
            tf.summary.scalar('bbvert_loss_l2', bbvert_loss_l2)
            tf.summary.scalar('bbvert_loss_ce', bbvert_loss_ce)
            tf.summary.scalar('bbvert_loss_iou', bbvert_loss_iou)
            tf.summary.scalar('bbscore_loss', bbscore_loss)

        with tf.variable_scope('pmask'):
            pmask_loss = Ops.get_loss_pmask(X_pc, y_pmask_pred, Y_pmask)
            tf.summary.scalar('pmask_loss', pmask_loss)

        return bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou, bbscore_loss, pmask_loss

    def build_graph(self, batch_size=1, num_gpus=1, pretrained_weight=None):
        """Build TensorFlow Graph

        Args:
            batch_size (int, optional): Batch size which can be divided with num_gpus. Defaults to 1.
            num_gpus (int, optional): Number of GPUs to use. Defaults to 1.
            pretrained_weight (str, optional): Path to pretrained weight (*.ckpt). If None, initialize weights by initializer. Defaults to None.
        """
        assert batch_size % num_gpus == 0, 'batch_size % num_gpus != 0'
        device_batch_size = int(batch_size / num_gpus)

        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                # 1. define inputs
                self.X_pc = tf.placeholder(shape=[None, None, self.points_cc], dtype=tf.float32, name='X_pc')
                self.Y_bbvert = tf.placeholder(shape=[None, self.bb_num, 2, 3], dtype=tf.float32, name='Y_bbvert')
                self.Y_pmask = tf.placeholder(shape=[None, self.bb_num, None], dtype=tf.float32, name='Y_pmask')
                self.lr = tf.placeholder(dtype=tf.float32, name='lr')

                # 2. define optimizers
                self.adam = tf.train.AdamOptimizer(learning_rate=self.lr)

                # -------------------------------------------
                # Get model and loss on multiple GPU devices
                # -------------------------------------------
                # Allocating variables on CPU first will greatly accelerate multi-gpu training.
                # Ref: https://github.com/kuza55/keras-extras/issues/21
                self.get_model(self.X_pc, self.Y_bbvert)

                # variables for training
                var_backbone = [var for var in tf.trainable_variables() if var.name.startswith('backbone') and not var.name.startswith('backbone/sem')]
                var_bbox = [var for var in tf.trainable_variables() if var.name.startswith('bbox')]
                var_pmask = [var for var in tf.trainable_variables() if var.name.startswith('pmask')]
                var_list = var_bbox + var_pmask + var_backbone

                # model outputs
                y_bbvert_pred_raw = []
                y_bbscore_pred_raw = []
                y_bbvert_pred = []
                pred_bborder = []
                y_bbscore_pred = []
                y_pmask_pred = []
                y_pmask_pred_raw = []

                # losses        
                bbvert_loss_all = []
                bbvert_loss_l2_all = []
                bbvert_loss_ce_all = []
                bbvert_loss_iou_all = []
                bbscore_loss_all = []
                pmask_loss_all = []
                total_loss_all = []

                # gradients from GPUs
                tower_grads = []

                for i in range(num_gpus):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        with tf.device('/gpu:%d' % (i)), tf.name_scope('gpu_%d' % (i)) as scope:
                            # Evenly split input data to each GPU
                            X_pc_batch = tf.slice(self.X_pc, [i*device_batch_size, 0, 0], [device_batch_size, -1, -1])
                            Y_bbvert_batch = tf.slice(self.Y_bbvert, [i*device_batch_size, 0, 0, 0], [device_batch_size, -1, -1, -1])
                            Y_pmask_batch = tf.slice(self.Y_pmask, [i*device_batch_size, 0, 0], [device_batch_size, -1, -1])

                            # model outputs
                            y_bbvert_pred_raw_batch, y_bbscore_pred_raw_batch, y_bbvert_pred_batch, pred_bborder_batch, y_bbscore_pred_batch, y_pmask_pred_batch, y_pmask_pred_raw_batch = \
                                self.get_model(X_pc_batch, Y_bbvert_batch)
                            
                            y_bbvert_pred_raw.append(y_bbvert_pred_raw_batch)
                            y_bbscore_pred_raw.append(y_bbscore_pred_raw_batch)
                            y_bbvert_pred.append(y_bbvert_pred_batch)
                            pred_bborder.append(pred_bborder_batch)
                            y_bbscore_pred.append(y_bbscore_pred_batch)
                            y_pmask_pred.append(y_pmask_pred_batch)
                            y_pmask_pred_raw.append(y_pmask_pred_raw_batch)

                            # losses
                            bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou, bbscore_loss, pmask_loss = \
                                self.get_loss(X_pc_batch, Y_bbvert_batch, Y_pmask_batch, y_bbvert_pred_batch, y_bbscore_pred_batch, y_pmask_pred_batch)
                                                        
                            total_loss = bbvert_loss + bbscore_loss + pmask_loss
                            tf.summary.scalar("total_loss", total_loss)

                            bbvert_loss_all.append(bbvert_loss)
                            bbvert_loss_l2_all.append(bbvert_loss_l2)
                            bbvert_loss_ce_all.append(bbvert_loss_ce)
                            bbvert_loss_iou_all.append(bbvert_loss_iou)
                            bbscore_loss_all.append(bbscore_loss)
                            pmask_loss_all.append(pmask_loss)
                            total_loss_all.append(total_loss)

                            # back propagation
                            grads = self.adam.compute_gradients(total_loss, var_list=var_list)
                            tower_grads.append(grads)                            

                # Merge pred and losses from multiple GPUs
                self.y_bbvert_pred_raw = tf.concat(y_bbvert_pred_raw, 0)
                self.y_bbscore_pred_raw = tf.concat(y_bbscore_pred_raw, 0)
                self.y_bbvert_pred = tf.concat(y_bbvert_pred, 0)
                self.pred_bborder = tf.concat(pred_bborder, 0)
                self.y_bbscore_pred = tf.concat(y_bbscore_pred, 0)
                self.y_pmask_pred = tf.concat(y_pmask_pred, 0)
                self.y_pmask_pred_raw = tf.concat(y_pmask_pred_raw, 0)

                self.bbvert_loss = tf.reduce_mean(bbvert_loss_all)
                self.bbvert_loss_l2 = tf.reduce_mean(bbvert_loss_l2_all)
                self.bbvert_loss_ce = tf.reduce_mean(bbvert_loss_ce_all)
                self.bbvert_loss_iou = tf.reduce_mean(bbvert_loss_iou_all)
                self.bbscore_loss = tf.reduce_mean(bbscore_loss_all)
                self.pmask_loss = tf.reduce_mean(pmask_loss_all)
                self.total_loss = tf.reduce_mean(total_loss_all)

                # Get training operator
                grads = average_gradients(tower_grads)
                self.optim = self.adam.apply_gradients(grads)
            
            print("Number of variables:", Ops.variable_count())
            
            # Session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)

            # Summary
            self.sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, self.sess.graph)
            self.sum_write_test = tf.summary.FileWriter(self.test_sum_dir)
            self.sum_merged = tf.summary.merge_all()

            # Restore/initialize model
            self.saver = tf.train.Saver(max_to_keep=1000)

            if pretrained_weight is not None and os.path.isfile(pretrained_weight + '.data-00000-of-00001'):
                print("Restoring saved model:", pretrained_weight)
                self.saver.restore(self.sess, pretrained_weight)
            else:
                print("Model not found, all weights are initialized")
                self.sess.run(tf.global_variables_initializer())
                    
