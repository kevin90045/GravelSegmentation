import itertools
from concurrent import futures

import numpy as np


class EvaluationLogger:
    def __init__(self, filepath: str, mode: str):
        self.file = open(filepath, mode)
        self.items = []

    def writeline(self, content: dict):
        # init items if empty
        if len(self.items) == 0:
            self.init_items(content)
        
        # get output data
        line = ""
        for key in self.items:
            line += str(content[key]) + ","
        line += "\n"

        # write
        self.file.write(line)
        self.file.flush()
    
    def init_items(self, content: dict):
        for key in content:
            self.items.append(key)
        
        self.items = sorted(self.items)

        for item in self.items:
            self.file.write(item + ",")
        self.file.write("\n")
        
        print("EvaluationLogger >> Init items.")

    def close(self):
        self.file.close()
        

class Evaluator:
    """Scene Evaluator
    """
    def __init__(self,
                 scene_points: np.ndarray,
                 gt_labels: np.ndarray,
                 iou_threshold=0.5,
                 use_multiprocessing=True,
                 num_workers=1):
        """Create Evaluator

        Args:
            scene_points (np.ndarray): Scene point data. (N, C)
            gt_labels (np.ndarray): Ground truth labels of each point. (N,)
            iou_threshold (float, optional): IoU threshold for evaluation. Defaults to 0.5.
            use_multiprocessing (bool, optional): If True, use multiprocessing evaluation. Defaults to True.
            num_workers (int, optional): Number of workers used in evaluation. Ignored if use_multiprocessing = False. Defaults to 1.
        """
        assert len(scene_points) == len(gt_labels), 'points number of scene points and gt labels are not the same'

        # Get ground truth info
        self.pts = scene_points[:, 0:3]
        self.gt = gt_labels.flatten()
        self.gt_ins_labels, self.gt_ins_pt_ind, self.gt_ins_pt_counts = self.get_ins_indices(self.gt)
        self.gt_ins_num = len(self.gt_ins_pt_ind)
        print('gt instances num:', self.gt_ins_num)

        # get bbox of gt
        self.bbox_margin = 0.01
        self.gt_bboxes = []
        for pt_ind in self.gt_ins_pt_ind:
            self.gt_bboxes.append(
                self.get_bbox(self.pts[pt_ind, 0:3], margin=self.bbox_margin))

        # pool for multiprocessing
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers if num_workers is not None else 1

        # metrics threshold
        self.iou_threshold = iou_threshold

    def get_ins_indices(self, labels: np.ndarray) -> tuple:
        """Get instance info

        Args:
            labels (np.ndarray): Labels of instances. (N1,)

        Returns:
            tuple (np.ndarray, list, np.ndarray): Tuple of unique labels (N2,), point indices of each instance (list of np.ndarray), and point numbers of each instance (N2,).
                Note that the labels < 0 will be ignored.
        """
        ins_labels, pt_ins_ind, ins_pt_counts = np.unique(labels, return_inverse=True, return_counts=True, axis=0)
        ins_pt_ind = np.split(np.argsort(pt_ins_ind), np.cumsum(ins_pt_counts[:-1]))

        kept_cond = ins_labels >= 0 # only consider labels > 0
        ins_labels = ins_labels[kept_cond]
        ins_pt_counts = ins_pt_counts[kept_cond]
        ins_pt_ind = list(np.sort(ins_pt_ind[i]) for i, cond in enumerate(kept_cond) if cond)
        assert len(ins_labels) == len(ins_pt_ind), 'get_ins_indices: len(ins_labels) == len(ins_pt_ind)'

        return ins_labels, ins_pt_ind, ins_pt_counts

    def get_bbox(self, cloud: np.ndarray, margin=0.0) -> float:
        """Get 3D bounding box

        Args:
            cloud (np.ndarray): Input point cloud. (N, 3).
            margin (float, optional): Margin to extend output output bounding box. Defaults to 0.

        Returns:
            np.ndarray: 3D bounding box (6, ) which is [min_x, min_y, min_z, max_x, max_y, max_z].
        """
        return np.concatenate(
            [cloud.min(axis=0) - margin,
             cloud.max(axis=0) + margin], axis=0)

    @staticmethod
    def get_bbox_iou(bboxes: tuple) -> float:
        """Get IoU between two bounding boxes

        Args:
            bboxes (tuple): Tuple of two bounding boxes from get_bbox.

        Returns:
            float: IoU.
        """
        bbox_p = bboxes[0]  # [minx, miny, minz, maxx, maxy, maxz]
        bbox_g = bboxes[1]  # [minx, miny, minz, maxx, maxy, maxz]

        volume_p = (bbox_p[3] - bbox_p[0]) * (bbox_p[4] - bbox_p[1]) * (bbox_p[5] - bbox_p[2])
        volume_g = (bbox_g[3] - bbox_g[0]) * (bbox_g[4] - bbox_g[1]) * (bbox_g[5] - bbox_g[2])

        bbox_pt = np.concatenate([bbox_p.reshape([1, -1]), bbox_g.reshape([1, -1])], axis=0)
        max_pt = np.min(bbox_pt[:, 3:6], axis=0)
        min_pt = np.max(bbox_pt[:, 0:3], axis=0)

        intersection = (max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1]) * (max_pt[2] - min_pt[2])
        union = volume_p + volume_g - intersection
        iou_tp = float(intersection) / (float(union) + 1e-8)

        return iou_tp

    def get_multi_bbox_iou(self, pr_bboxes: list) -> np.ndarray:
        """Get IoUs between inputs and ground truth bounding boxes

        Args:
            pr_bboxes (list): List of bounding boxes from get_bbox.

        Returns:
            np.ndarray: IoUs with size is (number of pr_bboxes, number of gt_bboxes).
        """
        pr_gt_bbox_pairs = list(itertools.product(pr_bboxes, self.gt_bboxes))
        
        if self.use_multiprocessing:
            chunksize = len(pr_gt_bbox_pairs) // self.num_workers
            with futures.ProcessPoolExecutor(self.num_workers) as pool:
                ious_bbox = list(pool.map(Evaluator.get_bbox_iou, pr_gt_bbox_pairs, chunksize=chunksize))
        else:
            ious_bbox = []
            for pr_gt in pr_gt_bbox_pairs:
                ious_bbox.append(Evaluator.get_bbox_iou(pr_gt))
        
        ious_bbox = np.array(ious_bbox)
        ious_bbox = np.reshape(ious_bbox, [len(pr_bboxes), self.gt_ins_num])

        return ious_bbox

    @staticmethod
    def get_intersect_point_indices(inputs: tuple) -> list:
        """https://stackoverflow.com/questions/59656759/what-is-a-best-way-to-intersect-multiple-arrays-with-numpy-array

        Args:
            inputs (tuple): Tuple of (point indices of predicted instance, list of point indices of ground truth instances, and a np.ndarray of point numbers of ground truth instances)

        Returns:
            list: List of intersection point indices between predicted instance and input ground truth instances.
        """
        ins_p = inputs[0]
        gt_ins_pt_ind = inputs[1]
        ins_g = np.concatenate(gt_ins_pt_ind)
        ins_g_pt_counts = inputs[2]

        inserted_ind = np.searchsorted(ins_p, ins_g)
        inserted_ind[inserted_ind == len(ins_p)] = 0
        mask_is_in = ins_p[inserted_ind] == ins_g

        splited_mask_is_in = np.split(mask_is_in, ins_g_pt_counts.cumsum())
        intersect_pt_ind = list(
            pt_ind[mask]
            for pt_ind, mask in zip(gt_ins_pt_ind, splited_mask_is_in))

        return intersect_pt_ind

    def get_multi_intersections(self, pr_ins_pt_ind: list, bbox_ious=None) -> np.ndarray:
        """Get number of intersection points between instances

        Args:
            pr_ins_pt_ind (list): List of point indices of instances.
            bbox_ious (np.ndarray, optional): IoUs between input instances and ground truth instances. size = (number of pr_bboxes, number of gt_bboxes).
                If provide, it can help filter instances and speed up computation. Defaults to None.

        Returns:
            np.ndarray: Number of intersection points between instances. size = (number of pr_bboxes, number of gt_bboxes).
        """
        # check bbox iou
        if bbox_ious is None:
            iou_cond = np.full((len(pr_ins_pt_ind), self.gt_ins_num), True, np.bool)
        else:
            iou_cond = bbox_ious > 0

        # prepare inputs
        inputs = []
        for pr_iou_cond, pr_pt_ind in zip(iou_cond, pr_ins_pt_ind):
            input_gt = list(itertools.compress(self.gt_ins_pt_ind, pr_iou_cond))
            input_gt_split_lengths = self.gt_ins_pt_counts[pr_iou_cond]
            inputs.append((pr_pt_ind, input_gt, input_gt_split_lengths))

        # compute intersection
        if self.use_multiprocessing:
            chunksize = len(inputs) // self.num_workers
            with futures.ProcessPoolExecutor(self.num_workers) as pool:
                intersect_ind = list(pool.map(Evaluator.get_intersect_point_indices, inputs, chunksize=chunksize))
        
        else:
            intersect_ind = []
            for i in inputs:
                intersect_ind.append(Evaluator.get_intersect_point_indices(i))
        
        intersect_ind = list(itertools.chain.from_iterable(intersect_ind)) # to 1-d list

        # get intersection points number
        intersection = np.zeros((len(pr_ins_pt_ind), self.gt_ins_num), dtype=np.int32)
        active_ind = np.argwhere(iou_cond)
        intersection[tuple(active_ind.T)] = np.array([len(ind) for ind in intersect_ind])

        return intersection

    def get_metrics(self, intersection: np.ndarray, pr_ins_pt_counts: np.ndarray) -> tuple:
        """Get evaluation metrics results: Precision, Recall, F1-score

        Args:
            intersection (np.ndarray): Number of intersection points. size = (number of predicted instances, number of ground truth instances).
            pr_ins_pt_counts (np.ndarray): Point numbers of each predicted instance.

        Returns:
            tuple (float, float, float): A tuple of Precision, Recall, F1-score
        """
        # number of predicted instances
        pr_ins_num = len(pr_ins_pt_counts)

        # compute IoUs
        unions = np.tile(np.reshape(self.gt_ins_pt_counts, [1, -1]), [pr_ins_num, 1]) + \
            np.tile(np.reshape(pr_ins_pt_counts, [-1, 1]), [1, self.gt_ins_num]) - intersection
        ious = intersection / unions

        # IoU >= threshold -> True Positive
        iou_maxs = ious.max(axis=1)
        tp = np.sum(iou_maxs >= self.iou_threshold)
        fp = pr_ins_num - tp

        # compute metrics
        precision = float(tp) / (tp + fp + 1e-8)
        recall = float(tp) / (self.gt_ins_num + 1e-8)
        f1_score = (2.0 * precision * recall) / (precision + recall + 1e-8)

        return precision, recall, f1_score

    def evaluate(self, pred_labels: np.ndarray) -> dict:
        """Evaluate predicted labels

        Args:
            pred_labels (np.ndarray): Predicted labels which size is (number of points,)

        Returns:
            dict: Evaluation results which includes 'precision', 'recall', and 'f1_score'.
        """
        assert len(pred_labels) == len(self.gt), 'The lengths of pred and gt labels are not the same'
        pred = pred_labels.flatten()

        # get each instance's point indices, numbers, and bbox
        _, pr_ins_pt_ind, pr_ins_pt_counts = self.get_ins_indices(pred)
        pr_ins_num = len(pr_ins_pt_ind)
        print('pred instances num:', pr_ins_num)
        pr_bboxes = []
        for pt_ind in pr_ins_pt_ind:
            pr_bboxes.append(self.get_bbox(self.pts[pt_ind, 0:3], margin=self.bbox_margin))

        # compute bbox ious
        bbox_ious = self.get_multi_bbox_iou(pr_bboxes)

        # in order to reduce computation
        # only compute point2point intersections for input that have bbox iou > 0
        intersection = self.get_multi_intersections(pr_ins_pt_ind, bbox_ious=bbox_ious)

        # compute precision, recall, and f1-score
        precision, recall, f1_score = self.get_metrics(intersection, pr_ins_pt_counts)

        results = dict()
        results["precision"] = precision
        results["recall"] = recall
        results["f1_score"] = f1_score
        
        return results
