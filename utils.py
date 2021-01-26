import numpy as np
from sklearn.neighbors import KDTree


def print_args(args):
    """Print arguments

    Args:
        args: The output of parse_args().
    """
    print("Input Arguments:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print()


def save_args(output_filename, args):
    """Save arguments as text file

    Args:
        output_filename (str): Output filename.
        args: The output of parse_args().
    """
    print("Input Arguments:")
    fout = open(output_filename, 'w')
    for arg in vars(args):
        fout.write("{}: {}\n".format(arg, getattr(args, arg)))
    fout.close()


def random_color(num_colors=1) -> np.ndarray:
    """Generate random colors

    Args:
        num_colors (int, optional): Number of colors to be generated. Defaults to 1.

    Returns:
        np.ndarray: Colors with size = (num_colors, 3).
    """
    np.random.seed(num_colors)
    return (np.random.random((num_colors, 3)) * 255).astype(np.uint8)


def get_scene_colors(ins_pred: np.ndarray) -> np.ndarray:
    """Get colors for each instance

    Args:
        ins_pred (np.ndarray): Instance labels. (N,).

    Returns:
        np.ndarray: Colors. (N, 3).
    """
    ins_pred_int = np.copy(ins_pred).astype(np.int)

    ins_labels, pt_ins_ind, ins_pt_counts = np.unique(ins_pred_int, return_inverse=True, return_counts=True, axis=0)
    ins_pt_ind = np.split(np.argsort(pt_ins_ind), np.cumsum(ins_pt_counts[:-1]))

    colors_all = np.zeros((ins_pred_int.shape[0], 3), dtype=np.uint8)
    colors = random_color(len(ins_labels))
    
    for i, pt_ind in enumerate(ins_pt_ind):
        colors_all[pt_ind, :] = colors[i, :]

    return colors_all


def fill_labels(pts_all: np.ndarray, labels: np.ndarray, verbose=False) -> np.ndarray:
    """Fill invalid label to closest instance label

    Args:
        pts_all (np.ndarray): Point XYZ. (N, 3).
        labels (np.ndarray): Instance labels. (N,)
        verbose (bool, optional): Display processing info or not. Defaults to False.

    Returns:
        np.ndarray: Fixed labels. (N,).
    """
    filled_labels = labels.copy()

    # Non-labeled: -1
    non_labeled_cond = filled_labels == -1
    non_labeled_num = non_labeled_cond.sum()
    if verbose:
        print('Non-labeled points num:', non_labeled_num)

    if non_labeled_num > 0:
        ins_labels = filled_labels[~non_labeled_cond]

        # KdTree finds the nearest labeled points
        kdt = KDTree(pts_all[~non_labeled_cond, 0:3], leaf_size=1, metric='euclidean')
        label_indices = kdt.query(pts_all[non_labeled_cond, 0:3], k=1, return_distance=False)

        # assign labels
        label_indices = np.reshape(label_indices, -1)
        filled_labels[non_labeled_cond] = ins_labels[label_indices]

    return filled_labels