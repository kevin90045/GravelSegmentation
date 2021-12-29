import glob
import os

import numpy as np
from tqdm import tqdm

from utils import get_scene_colors
from helper_ply import write_ply


# path = "/home/chen/Projects/Gravel/data/testing/model_v4.3_20210124_ep49/real_scenes/scenes_original/bin/manual_post_processing/asc/*.*"
# files = sorted(glob.glob(path))
# print(len(files), "files")

# data_all = []
# for i, f in tqdm(enumerate(files)):
#     xyz = np.loadtxt(f)[:,0:3]
#     labels = np.ones((len(xyz), 1), dtype=np.int32) * (i+1)
#     ins = np.concatenate([xyz, labels], axis=-1)
#     data_all.append(ins)

# data_all = np.concatenate(data_all, axis=0)
# np.save(
#     "/home/chen/Projects/Gravel/data/testing/model_v4.3_20210124_ep49/real_scenes/scenes_original/bin/manual_post_processing/manual.npy",
#     data_all)

# colors = get_scene_colors(data_all[:,-1])
# write_ply(
#     "/home/chen/Projects/Gravel/data/testing/model_v4.3_20210124_ep49/real_scenes/scenes_original/bin/manual_post_processing/manual.ply",
#     [data_all[:, 0:3], colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

data = np.load(
    "/home/chen/Projects/Gravel/data/testing/model_v4.3_20210124_ep49/real_scenes/scenes_original/bin/manual_post_processing/S4_filterbyhm_SOR_5NN_0SD75_labeled_0.001_blocks_v4.3_iter_0.9_50_10_0.0045_10_original_pred_manual.npy")
colors = get_scene_colors(data[:, -1])

# get point indices of each instance
ins_labels, pt_ins_ind, ins_pt_counts = np.unique(data[:,-1],
                                                    return_inverse=True,
                                                    return_counts=True,
                                                    axis=0)
ins_pt_ind = np.split(np.argsort(pt_ins_ind),
                        np.cumsum(ins_pt_counts[:-1]))

# save instances to PLY
print(
    "Saving separate instances to",
    "/home/chen/Projects/Gravel/data/testing/model_v4.3_20210124_ep49/real_scenes/scenes_original/bin/manual_post_processing/txt_rgb"
)
for label, pt_ind in tqdm(zip(ins_labels, ins_pt_ind)):
    if label == -1: continue

    output_ins_name = os.path.join(
        "/home/chen/Projects/Gravel/data/testing/model_v4.3_20210124_ep49/real_scenes/scenes_original/bin/manual_post_processing/txt_rgb",
        str(int(label)).zfill(4) + ".ply")
    write_ply(output_ins_name, [data[pt_ind, 0:3], colors[pt_ind, 0:3]],
                ['x', 'y', 'z', 'red', 'green', 'blue'])
