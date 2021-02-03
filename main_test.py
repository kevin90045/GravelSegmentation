import os
import argparse
from glob import glob
from os.path import join, splitext, basename, isdir, isfile, dirname, abspath, exists

from tqdm import tqdm
import numpy as np

from helper_data import DataUtils
from volume import Volume
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path to original scene file. Support h5, npy, xyz formats. Can be a file path or directory')
parser.add_argument('--pred', type=str, default=None, help='Path to prediction files from Tester. Can be a file path or directory')
parser.add_argument('--model', type=str, default=None, help='Path to model file (*.ckpt)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size during inference. Ignored when using Iterative Prediction')
parser.add_argument('--iter', help='Use Iterative Prediction', action="store_true")
parser.add_argument('--min_prob', type=float, default=0.8, help='Minimum probability score for Iterative Prediction')
parser.add_argument('--min_ins_pts_num', type=int, default=50, help='Minimum point number of instance for Iterative Prediction')
parser.add_argument('--max_iter', type=int, default=10, help='Maximum iteration for Iterative Prediction')
parser.add_argument('--save_pred', help='Save prediction results of blocks to HDF5 file', action="store_true")
parser.add_argument('--cell_size', type=float, default=5e-3, help='Cell size for BlockMerging')
parser.add_argument('--overlap_cell_thres', type=int, default=0, help='Threshlod for number of overlapping cells for BlockMerging')
parser.add_argument('--all', help='Assign label to all points with nearest neighbor', action="store_true")
parser.add_argument('--save_ply', help='Save predicted scene points with RGB colors to PLY format', action="store_true")
parser.add_argument('--save_ins', help='Save separate instance with unique RGB colors to PLY format', action="store_true")
parser.add_argument('--eval', help='Conduct evaluation', action="store_true")
parser.add_argument('--iou_thres', type=float, default=0.5, help='IoU threshold for evaluation')
parser.add_argument('--num_workers', type=int, default=None, help='Nunber of workers for multiprocessing evaluation')

ARGS = parser.parse_args()


PATH = ARGS.path
PRED = ARGS.pred
### TESTING
MODEL_PATH = ARGS.model
BATCH_SIZE = ARGS.batch_size
USE_ITER = ARGS.iter
MIN_PROB = ARGS.min_prob
MIN_INS_PTS_NUM = ARGS.min_ins_pts_num
MAX_ITER = ARGS.max_iter
SAVE_PRED = ARGS.save_pred
### BLOCK_MERGING
CELL_SIZE = ARGS.cell_size
OVERLAP_CELL_THRES = ARGS.overlap_cell_thres
SAVE_PLY = ARGS.save_ply
SAVE_INS = ARGS.save_ins
ASSIGN_ALL = ARGS.all
### EVALUATION
EVAL = ARGS.eval
IOU_THRES = ARGS.iou_thres
NUM_WORKERS = ARGS.num_workers

# import if needed
TESTER = None
PLY_WRITER = None
EVALUATOR = None
EVALUATION_LOGGER = None

if SAVE_PLY or SAVE_INS:
    try:
        from helper_ply import write_ply
        PLY_WRITER = write_ply
    except Exception as e:
        print(e)

if EVAL:
    try:
        from datetime import datetime
        from evaluator import Evaluator, EvaluationLogger
        EVALUATOR = Evaluator

        output_filename = "{}_{}_{}_{}_{}.csv".format(splitext(basename(PATH))[0], CELL_SIZE, OVERLAP_CELL_THRES, IOU_THRES, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        logger_name = join(dirname(PATH), output_filename)
        EVALUATION_LOGGER = EvaluationLogger(logger_name, "w")
    except Exception as e:
        print(e)


def get_filenames(path: str, exts=['h5']):
    """Get filenames with specific extentions

    Args:
        path (str): Path to file or directory.
        exts (list, optional): List of strings of extentions. Defaults to ['h5'].

    Returns:
        list: File paths.
    """
    try:
        if isdir(path):
            files = []
            for ext in exts:
                files.extend(glob(join(path, '*.' + ext)))
            return sorted(files)
        
        elif isfile(path):
            return [path]
        
        else:
            print("Cannot identify path:", path)
            return list()
    except:
        return list()


def get_input_pairs(filenames: list, predictions: list):
    """Get filename-prediction dict

    Args:
        filenames (list): File paths.
        predictions (list): Paths of prediction files. Notice that only the prediction filename that contains filename and 'blocks_pred' will be considered.

    Returns:
        dict: Key are filename, value is a list of predictions. If a file has no corresponding prediction file, the value will be None.
    """
    results = dict()

    if len(filenames) == len(predictions) == 1:
        results[filenames[0]] = [predictions[0]]
        return results

    # matching method: in default, the prediction files saved by Tester will be named "filename_blocks_pred.h5" 
    for filename in filenames:
        results[filename] = [None]

        search_str = splitext(basename(filename))[0]
        search_str = search_str.replace("_original", "")
        for pred in predictions:
            target_str = basename(pred)

            if search_str in target_str and "blocks_pred" in target_str:
                if len(results[filename]) == 1 and results[filename][0] is None:
                    results[filename] = [pred]
                else:
                    results[filename] += [pred]
    
    return results


def create_tester():
    """Create SimpleTester or IterativeTester

    Returns:
        SimpleTester | IterativeTester | None: If exception occurs, return None.
    """
    from tester import SimpleTester, IterativeTester
    from helper_data import DataConfigs

    configs = DataConfigs()
    try:
        if USE_ITER:
            tester = IterativeTester(MODEL_PATH, configs, MIN_PROB, MIN_INS_PTS_NUM, MAX_ITER)
        else:
            tester = SimpleTester(MODEL_PATH, configs, BATCH_SIZE)
    except:
        return None
    
    return tester


def block_merging(block_pts: list, block_ins_pred: list):
    """BlockMerging algorithm

    Args:
        block_pts (list): List of block points (N, 3). The 3 channels are scene normalized x, scene normalized y, scene normalized z.
        block_ins_pred (list): List of block predicted labels (N,).

    Returns:
        Volume: A merged volume.
    """
    print('BlockMerging: {}, {}'.format(CELL_SIZE, OVERLAP_CELL_THRES))        
    volume = Volume(CELL_SIZE, OVERLAP_CELL_THRES)
    for block, ins_pred in tqdm(zip(block_pts, block_ins_pred)):
        volume.block_merging(block, ins_pred)
    
    return volume


def write_ply(output_filename: str, xyz: np.ndarray, colors: np.ndarray):
    """Write PLY file

    Args:
        output_filename (str): Output filename.
        xyz (np.ndarray): Data to output (N, 3).
        colors (np.ndarray): RGB colors of each points (N, 3).
    """
    if PLY_WRITER is None:
        return
    
    PLY_WRITER(output_filename, [xyz, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
    print('Saved results to', output_filename)


def write_ins_ply(output_dir: str, xyz: np.ndarray, labels: np.ndarray, colors=None):
    """Write instances to PLY files

    Args:
        output_dir (str): Output directory.
        xyz (np.ndarray): All xyz. (N, 3).
        labels (np.ndarray): All labels. (N,).
        colors (np.ndarray, optional): RGB colors to xyz. (N, 3). if None, generate automatically. Defaults to None.
    """
    if PLY_WRITER is None:
        return
    
    assert len(xyz) == len(labels) == len(colors)
    
    # create directories if not exist
    if not exists(output_dir):
        os.makedirs(output_dir)
    
    # get point indices of each instance
    ins_labels, pt_ins_ind, ins_pt_counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
    ins_pt_ind = np.split(np.argsort(pt_ins_ind), np.cumsum(ins_pt_counts[:-1]))

    # generate random colors
    if colors is None:
        colors = utils.get_scene_colors(labels)

    # save instances to PLY
    print("Saving separate instances to", output_dir)
    for label, pt_ind in tqdm(zip(ins_labels, ins_pt_ind)):
        if label == -1: continue

        output_ins_name = join(output_dir, str(label).zfill(4) + ".ply")
        PLY_WRITER(output_ins_name, [xyz[pt_ind, 0:3], colors[pt_ind, 0:3]], ['x', 'y', 'z', 'red', 'green', 'blue'])


def eval(xyz: np.ndarray, ins_gt: np.ndarray, ins_pred: np.ndarray):
    """Process evaluation

    Args:
        xyz (np.ndarray): XYZ data. (N, 3).
        ins_gt (np.ndarray): Ground truth instance labels. (N,).
        ins_pred (np.ndarray): Predicted instance labels. (N,).

    Returns:
        dict: Evaluation results which includes 'precision', 'recall', and 'f1_score'.
    """
    if EVALUATOR is None:
        return

    print('Initializing Evaluator with IoU threshold:', IOU_THRES)
    evaluator = EVALUATOR(xyz, ins_gt, iou_threshold=IOU_THRES, num_workers=NUM_WORKERS)
    
    return evaluator.evaluate(ins_pred)


def process_original_scene(filename: str, pred_filename: str, output_dir='.'):
    print("Start original scene file processing")
    scene_data = DataUtils.load_scene_file(filename)

    if not SAVE_PLY and not SAVE_INS and not EVAL:
        print("No task to do. Use --save_ply, --save_ins or --eval")
        return

    # Load prediction file
    if pred_filename is None:
        print("No prediction file. Skip\n")
        return

    print("Prediction file:", pred_filename)
    block_pts, _, block_ins_pred = DataUtils.load_prediction_file(pred_filename)
    print("Number of blocks:", len(block_pts))
    
    # BlockMerging
    volume = block_merging(block_pts, block_ins_pred)
    
    # Get instance label of all points
    xyz_all = DataUtils.normalize_xyz(scene_data[:, 0:3])
    scene_ins_pred = volume.get_labels(xyz_all)
    scene_ins_colors = utils.get_scene_colors(scene_ins_pred)

    if ASSIGN_ALL:
        scene_ins_pred = utils.fill_labels(scene_data[:, 0:3], scene_ins_pred)
    
    labeled_cond = scene_ins_pred != -1
    unique_scene_ins_pred, pt_ins_ind, ins_pt_counts = np.unique(scene_ins_pred, return_inverse=True, return_counts=True, axis=0)
    num_ins_pred = np.sum(unique_scene_ins_pred != -1)

    print('Number of points:', len(scene_data))
    print('Number of predicted points:', len(scene_data[labeled_cond, :]))
    print('Number of predicted instances:', num_ins_pred)
    
    # Output info
    output_ply_basename = join(output_dir, basename(pred_filename).replace('_blocks_pred.h5', ''))
    output_ply_basename = "{}_{}_{}_original_pred".format(output_ply_basename, CELL_SIZE, OVERLAP_CELL_THRES)
    if ASSIGN_ALL:
        output_ply_basename += "_all" 

    # Save PLY
    if SAVE_PLY:
        write_ply(output_ply_basename + ".ply", scene_data[labeled_cond, 0:3], scene_ins_colors[labeled_cond, :])
    
    # Save separate instances
    if SAVE_INS:
        write_ins_ply(output_ply_basename, scene_data[:, 0:3], scene_ins_pred, colors=scene_ins_colors)
    
    # Evaluation
    if EVAL:
        metrics = eval(scene_data[:, 0:3], scene_data[:, -1], scene_ins_pred)
        print("Precision: {} / Recall: {} / F1-score: {}".format(metrics['precision'], metrics['recall'], metrics['f1_score']))

        if EVALUATION_LOGGER is not None:
            num_ins_gt = np.sum(np.unique(scene_data[:, -1]) != -1)
            log = {
                "file": filename,
                "pred_file": pred_filename,
                "num_pts": len(scene_data),
                "num_pts_pred": len(scene_data[labeled_cond, :]),
                "num_ins_pred": num_ins_pred,
                "num_ins_gt": num_ins_gt,
                "cell_size": CELL_SIZE,
                "overlap_cell_thres": OVERLAP_CELL_THRES,
                "iou_thres": IOU_THRES }
            
            EVALUATION_LOGGER.writeline({**log, **metrics})


def process_scene_blocks(filename, pred_filename, output_dir='.'):
    print("Start sub-sampled scene blocks processing")
    blocks, _ = DataUtils.load_sampled_file(filename)
    data = np.concatenate(blocks, axis=0)

    # Test if needed
    if pred_filename is not None:
        print("Prediction file:", pred_filename)
        block_pts, block_ins_gt, block_ins_pred = DataUtils.load_prediction_file(pred_filename)
    
    elif TESTER is not None:
        print("No prediction file. Starting testing...")
        pred_filename = splitext(filename)[0]

        if USE_ITER:
            print("Iteratively Prediction: {}, {}, {}".format(MIN_PROB, MIN_INS_PTS_NUM, MAX_ITER))
            pred_filename += "_iter_{}_{}_{}".format(MIN_PROB, MIN_INS_PTS_NUM, MAX_ITER)

        block_pts, block_ins_gt, block_ins_pred = TESTER.test(filename, SAVE_PRED, pred_filename)
    
    else:
        print("No Tester can be used. Skip\n")
        return

    print("Number of blocks:", len(block_pts))
    
    # Continue if no other tasks
    if not SAVE_PLY and not SAVE_INS and not EVAL:
        return
    
    # BlockMerging
    volume = block_merging(block_pts, block_ins_pred)

    # Get instance label of all points
    scene_ins_pred = volume.get_labels(data[:, 3:6])
    scene_ins_colors = utils.get_scene_colors(scene_ins_pred)
    num_ins_pred = np.sum(np.unique(scene_ins_pred) != -1)

    print('Number of points:', len(data))
    print('Number of predicted instances:', num_ins_pred)
    
    # Output info
    output_ply_basename = join(output_dir, basename(pred_filename).replace('_blocks_pred.h5', ''))
    output_ply_basename = "{}_{}_{}_pred".format(output_ply_basename, CELL_SIZE, OVERLAP_CELL_THRES)

    # Save PLY
    if SAVE_PLY:
        write_ply(output_ply_basename + ".ply", data[:, 0:3], scene_ins_colors)
    
    # Save separate instances
    if SAVE_INS:
        write_ins_ply(output_ply_basename, data[:, 0:3], scene_ins_pred, colors=scene_ins_colors)

    # Evaluation
    if EVAL:
        pts_all = np.concatenate(block_pts, axis=0)
        ins_gt_all = np.concatenate(block_ins_gt, axis=0)
        ins_pred_all = volume.get_labels(pts_all)        

        metrics = eval(pts_all, ins_gt_all, ins_pred_all)
        print("Precision: {} / Recall: {} / F1-score: {}".format(metrics['precision'], metrics['recall'], metrics['f1_score']))

        if EVALUATION_LOGGER is not None:
            num_ins_gt = np.sum(np.unique(ins_gt_all) != -1)
            log = {
                "file": filename,
                "pred_file": pred_filename,
                "num_pts": len(data),
                "num_pts_pred": len(data),
                "num_ins_pred": num_ins_pred,
                "num_ins_gt": num_ins_gt,
                "cell_size": CELL_SIZE,
                "overlap_cell_thres": OVERLAP_CELL_THRES,
                "iou_thres": IOU_THRES }
            
            EVALUATION_LOGGER.writeline({**log, **metrics})



if __name__ == "__main__":
    utils.print_args(ARGS)

    # Get filenames
    filenames = get_filenames(PATH, exts=['h5','npy','xyz'])

    if len(filenames) == 0:
        print("No available file at {}".format(PATH))
        exit()

    # predictions = get_predictions(filenames, get_filenames(PRED, exts=['h5']))
    predictions = get_input_pairs(filenames, get_filenames(PRED, exts=['h5']))

    # Create tester is needed
    values_all = [item for values in predictions.values() for item in values]
    if None in values_all:
        print("Create tester")
        TESTER = create_tester()
    
        if TESTER is None:
            print("\nCannot load model. The files need inference will be skipped\n")
    
    # Start processing
    print("Total number of files: {}\n".format(len(filenames)))

    for i, filename in enumerate(filenames):
        print("[{}/{}]".format(i+1, len(filenames)))
        
        output_dir = dirname(abspath(filename))
        pred_filenames = predictions[filename]

        print("Processing", filename, ",", len(pred_filenames), "pred files")

        # Check scene file type: original or sub-sampled blocks
        # original
        try:
            DataUtils.load_scene_file(filename)
            for pred_filename in pred_filenames:
                process_original_scene(filename, pred_filename, output_dir=output_dir)
        # sub-sampled blocks
        except:
            try:
                DataUtils.load_sampled_file(filename)
                for pred_filename in pred_filenames:
                    process_scene_blocks(filename, pred_filename, output_dir=output_dir)
            except Exception as e:
                print(e)
                continue
        
        print()
    
    if EVALUATION_LOGGER is not None:
        EVALUATION_LOGGER.close()

    print("Finished")
