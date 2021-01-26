import os
import sys
import glob
import shutil

PATH = "/home/chen/Projects/Gravel/data/data_gravel_v4.3/val"

if __name__ == "__main__":
    # output dir
    root_dir = PATH + "_" + str(sys.argv[1]).zfill(2)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    files = glob.glob(os.path.join(PATH, "*.h5"))
    
    for f in files:
        if "blocks_pred" in f:
            output_filename = os.path.basename(f)
            output_filename = os.path.join(root_dir, output_filename)
            shutil.move(f, output_filename)