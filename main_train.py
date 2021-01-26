import os
import math
import shutil
import argparse
import multiprocessing as mp

from BoNet import BoNet
from helper_data import DataLoader, DataConfigs
from utils import print_args, save_args


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='The root path of dataset')
parser.add_argument('--train_dir', type=str, nargs="+", default=["train"], help='Folder names for training')
parser.add_argument('--val_dir', type=str, nargs="+", default=["val"], help='Folder names for validation during training')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--batch_size', type=int, default=24, help='Training/Validation batch size')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='Initial learning rate for Adam optimizer')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='Index of starting epoch')
parser.add_argument('--logdir', type=str, default="log", help='Directory for logging summaries and checkpoints')
parser.add_argument('--retrain', action="store_true", help='Keep logdir if exists')
parser.add_argument('--pretrained_weight', type=str, default=None, help='Path to pretrained model file (*.ckpt)')
parser.add_argument('--num_workers', type=int, default=1, help='Nunber of workers for multiprocessing data loader')

ARGS = parser.parse_args()

# DATA
DATASET_PATH = ARGS.dataset_path
TRAIN_DIR = ARGS.train_dir
VAL_DIR = ARGS.val_dir
NUM_WORKERS = ARGS.num_workers
# TRAINING
NUM_GPUS = ARGS.num_gpus
BATCH_SIZE = ARGS.batch_size
NUM_EPOCHS = ARGS.num_epochs
START_EPOCH = ARGS.start_epoch
LOG_DIR = ARGS.logdir
RETRAIN = ARGS.retrain
PRETRAINED_WEIGHT = ARGS.pretrained_weight
INIT_LRATE = ARGS.learning_rate
L_RATE_DROP = 0.5
L_RATE_EPOCHS_DROP = 10


def learning_rate_step_decay(epoch):
    initial_lrate = INIT_LRATE
    drop = L_RATE_DROP
    epochs_drop = L_RATE_EPOCHS_DROP
    lrate = initial_lrate * math.pow(drop,  math.floor((1+epoch)/epochs_drop))
    return lrate


def train(net: BoNet, train_data: DataLoader, test_data: DataLoader):
    train_data.init_data_pipeline()
    test_data.init_data_pipeline()

    for ep in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        # learning rate
        l_rate = learning_rate_step_decay(ep)
        print('learning rate:', l_rate)
        
        # total train batch number
        total_train_batch_num = train_data.total_batch_num
        print('total train batch num:', total_train_batch_num)

        for i in range(total_train_batch_num):
            # get training data
            bat_pc, _, bat_bbvert, bat_pmask = train_data.get_batch()

            # check batch size because the last batch may not be split to multiple GPUs
            if len(bat_pc) % NUM_GPUS != 0:
                print("bat_pc :", bat_pc.shape)
                continue
            
            # optimization
            _, ls_bbvert, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask, ls_total = net.sess.run(
                [
                    net.optim, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou, net.bbscore_loss, net.pmask_loss, net.total_loss
                ],
                feed_dict={
                    net.X_pc: bat_pc[:, :, 0:6],
                    net.Y_bbvert: bat_bbvert,
                    net.Y_pmask: bat_pmask,
                    net.lr: l_rate
                })

            # training summary
            if i != 0 and (i % 200 == 0 or i == total_train_batch_num - 1):
                sum_train = net.sess.run(
                    net.sum_merged,
                    feed_dict={
                        net.X_pc: bat_pc[:, :, 0:6],
                        net.Y_bbvert: bat_bbvert,
                        net.Y_pmask: bat_pmask,
                        net.lr: l_rate
                    })
                net.sum_writer_train.add_summary(sum_train, ep * total_train_batch_num + i)

            print('ep', ep, 'i', i, 'bbvert', ls_bbvert, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'iou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask, 'total', ls_total)

            # random testing and summary
            if i != 0 and (i % 200 == 0 or i == total_train_batch_num - 1):
                bat_pc, _, bat_bbvert, bat_pmask = test_data.get_batch()
                ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask, sum_test, pred_bborder = net.sess.run(
                    [
                        net.bbvert_loss, net.bbvert_loss_l2,
                        net.bbvert_loss_ce, net.bbvert_loss_iou,
                        net.bbscore_loss, net.pmask_loss, net.sum_merged, net.pred_bborder
                    ],
                    feed_dict={
                        net.X_pc: bat_pc[:, :, 0:6],
                        net.Y_bbvert: bat_bbvert,
                        net.Y_pmask: bat_pmaskrain: False
                    })
                net.sum_write_test.add_summary(sum_test, ep * total_train_batch_num + i)
                print('ep', ep, 'i', i, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'iou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)
                print('test pred bborder', pred_bborder)

        # saving model
        net.saver.save(net.sess, save_path=os.path.join(net.train_mod_dir, 'model' + str(ep).zfill(3) + '.ckpt'))


if __name__ == '__main__':
    print_args(ARGS)

    # Create 3D-BoNet    
    configs = DataConfigs()
    net = BoNet(configs=configs)
    net.creat_folders(name=LOG_DIR, re_train=RETRAIN)
    net.build_graph(BATCH_SIZE, NUM_GPUS, pretrained_weight=PRETRAINED_WEIGHT)
    
    # Backup files
    shutil.copyfile("main_train.py", os.path.join(LOG_DIR, "main_train.py"))
    shutil.copyfile("helper_data.py", os.path.join(LOG_DIR, "helper_data.py"))
    save_args(os.path.join(LOG_DIR, "arguments.txt"), ARGS)

    # Create DataLoader
    if NUM_WORKERS is None: NUM_WORKERS = 1
    train_data = DataLoader(DATASET_PATH, TRAIN_DIR, batch_size=BATCH_SIZE, num_works=NUM_WORKERS, epoch=NUM_EPOCHS)
    val_data = DataLoader(DATASET_PATH, VAL_DIR, batch_size=BATCH_SIZE, num_works=NUM_WORKERS, epoch=NUM_EPOCHS)
    
    # Training
    train(net, train_data, val_data)
