import numpy as np

class Volume:
    """A volume to conduct BloclMerging algorithm
    """
    def __init__(self, cell_size=5e-3, overlap_cell_thres=0):
        """Initialize Volume

        Args:
            cell_size (float, optional): Cell size of volume. Defaults to 5e-3.
            overlap_cell_thres (int, optional): Threshold for number of overlapping cells when BlockMerging. Defaults to 0.
        """
        self.cell_size = cell_size
        self.overlap_cell_thres = overlap_cell_thres

        self.volume = self.create_volume(self.cell_size)
        self.curr_max_label = -1
    
    def create_volume(self, cell_size=5e-3) -> np.ndarray:
        """Create volume

        Args:
            cell_size (float, optional): Cell size of volume. Defaults to 5e-3.

        Returns:
            np.ndarray: A volume whose width = int(1/cell_size)+2
        """
        volume_num = int(1. / cell_size) + 2
        volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
        return volume
    
    def block_merging(self, pts: np.ndarray, labels: np.ndarray):
        """BlockMerging algorithm

        Args:
            pts (np.ndarray): Point XYZ (N, 3). Notice that the XYZ should be normalized to [0, 1] first.
            labels (np.ndarray): Point instance labels (N,).
        """
        # only consider labeled points
        labels = labels.ravel()
        labeled_cond = np.flatnonzero(labels != -1)

        valid_labels = labels[labeled_cond]
        xyz_int = (pts[labeled_cond, 0:3] / self.cell_size).astype(np.int32)

        # get unique labels and corresponding point indices
        unique_labels, pt_label_ind, label_counts = np.unique(valid_labels, return_inverse=True, return_counts=True)
        label_pt_ind = np.split(np.argsort(pt_label_ind), np.cumsum(label_counts[:-1]))

        # get overlapping labels in volume of each unique label
        if self.curr_max_label is None:
            self.curr_max_label = np.max(self.volume)
        
        # get overlapping labels in volume of each unique label
        volume_labels_all = self.volume[tuple(xyz_int.T)]
        xyz_int_ind = np.flatnonzero(volume_labels_all == -1) # empty cells indice

        if xyz_int_ind.shape[0] == 0: # retrun if no empty cells
            return

        overlap_label_counts = np.zeros([len(unique_labels), self.curr_max_label + 2], dtype=np.int32)

        for i, pt_ind in enumerate(label_pt_ind):
            volume_labels = volume_labels_all[pt_ind]
            volume_labels = volume_labels[np.where(volume_labels != -1)] # exclude non-labeled cell
            unique_volume_labels, volume_label_counts = np.unique(volume_labels, return_counts=True)
            overlap_label_counts[i, unique_volume_labels] += volume_label_counts

        # get max-overlapping labels in volume
        new_labels = np.argmax(overlap_label_counts, axis=1)
        max_overlap_label_counts = overlap_label_counts[np.arange(len(unique_labels)), new_labels]

        # assign labels for new instances
        new_label_ind = np.flatnonzero((max_overlap_label_counts <= self.overlap_cell_thres) & (label_counts > 0))
        new_labels[new_label_ind] = np.arange(1, len(new_label_ind) + 1) + self.curr_max_label
        
        # get unique volume indice of points
        cells, pt_cell_ind, cell_counts = np.unique(xyz_int[xyz_int_ind, :], return_inverse=True, return_counts=True, axis=0)
        cell_pt_ind = np.split(np.argsort(pt_cell_ind), np.cumsum(cell_counts[:-1]))

        # update volume: the label of a cell in volume will be the label of the point with smallest index
        # print("v max_0", np.max(volume))
        unique_label_ind = []
        for pt_ind in cell_pt_ind:
            min_pt_idx = np.min(pt_ind)
            pt_idx = xyz_int_ind[min_pt_idx] # the original index of point
            unique_label_ind.append(pt_label_ind[pt_idx])
        self.volume[tuple(cells.T)] = new_labels[unique_label_ind]

        # update max label
        self.curr_max_label = max(np.max(new_labels[unique_label_ind]), self.curr_max_label) 

    def get_labels(self, pts: np.ndarray) -> np.ndarray:
        """Get instance labels of input points in volume

        Args:
            pts (np.ndarray): Point XYZ (N, 3). Notice that the XYZ should be normalized to [0, 1] first.

        Returns:
            np.ndarray: Corresponding instance labels in the volume. (N,).
        """
        xyz_int = (pts / self.cell_size).astype(np.int32)
        return self.volume[tuple(xyz_int.T)]