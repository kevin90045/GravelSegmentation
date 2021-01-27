"""Modified from: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
"""
import os
import shutil
import sys
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from BoNet import BoNet
from helper_data import DataLoader, DataConfigs
from utils import print_args


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='The root path of dataset')
parser.add_argument('--train_dir', type=str, nargs="+", default=["train"], help='Folder names for training')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--batch_size', type=int, default=24, help='Training/Validation batch size')
parser.add_argument('--init_learning_rate', type=float, default=1e-8, help='Initial learning rate')
parser.add_argument('--final_learning_rate', type=float, default=10., help='Final learning rate')
parser.add_argument('--pretrained_weight', type=str, default=None, help='Path to pretrained model file (*.ckpt)')
parser.add_argument('--vis', help='Show plot', action="store_true")

ARGS = parser.parse_args()

# DATA
DATASET_PATH = ARGS.dataset_path
TRAIN_DIR = ARGS.train_dir
# TRAINING
NUM_GPUS = ARGS.num_gpus
BATCH_SIZE = ARGS.batch_size
INIT_LRATE = ARGS.init_learning_rate
FINAL_LRATE = ARGS.final_learning_rate
PRETRAINED_WEIGHT = ARGS.pretrained_weight
# VISUALIZATION
VIS = ARGS.vis


def find_lr(net: BoNet, data: DataLoader, init_value=1e-8, final_value=10., beta=0.98):
    data.init_data_pipeline()

    num = data.total_batch_num
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    print("total batch num:", num)
    for i in range(num):
        batch_num += 1
        print(i, "lr", lr, end=' ')

        # As before, get the loss for this mini-batch of inputs/outputs
        bat_pc, _, bat_bbvert, bat_pmask = data.get_batch()

        _, ls_bbvert_all, ls_bbscore, ls_pmask = net.sess.run(
            [
                net.optim, net.bbvert_loss, net.bbscore_loss, net.pmask_loss
            ],
            feed_dict={
                net.X_pc: bat_pc[:, :, 0:6],
                net.Y_bbvert: bat_bbvert,
                net.Y_pmask: bat_pmask,
                net.lr: lr
            })
        
        total_loss = ls_bbvert_all + ls_bbscore + ls_pmask

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * total_loss
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            print("Loss is exploding, Stop processing")
            return log_lrs, losses
        
        # Stop if the loss is exploding
        if batch_num > 1 and np.isnan(smoothed_loss):
            print("Loss is NaN, Stop processing")
            return log_lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        print("smoothed_loss", smoothed_loss, "best_loss", best_loss)

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Update the lr for the next step
        lr *= mult
    
    return log_lrs, losses


if __name__ == '__main__':
    print_args(ARGS)

    # Create 3D-BoNet
    configs = DataConfigs()
    net = BoNet(configs=configs)
    net.creat_folders(name="lr_finder", re_train=False)
    net.build_graph(BATCH_SIZE, NUM_GPUS, PRETRAINED_WEIGHT)

    # Data Loader
    data = DataLoader(DATASET_PATH, TRAIN_DIR, batch_size=BATCH_SIZE, num_works=1, epoch=1)
    
    # Learning rate finder
    log_lrs, losses = find_lr(net, data, init_value=INIT_LRATE, final_value=FINAL_LRATE, beta=0.98)
    shutil.rmtree("lr_finder")

    # Show all results
    plt.plot(log_lrs, losses)
    plt.savefig("lr_finder.png")
    if VIS:
        plt.show()
    
    # Skip of the first 10 values and the last 5 to focus on the interesting parts
    plt.close()
    plt.plot(log_lrs[10:-5], losses[10:-5])
    plt.savefig("lr_finder_clip.png")
    if VIS:
        plt.show()