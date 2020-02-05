from __future__ import division, print_function

from collections import defaultdict
import numpy as np
import json
import os
import cv2
from tqdm import tqdm

def read_json(jsonname):
    json_yolo = {}
    with open(jsonname) as f:
        data = f.readlines()
    for datum in data:
        datum = json.loads(datum)
        json_yolo[datum["filename"]] = datum["tag"]
    return json_yolo

def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

class VOCMApMetric():
    """
    Calculate mean AP for object detection task
    Parameters:
    ---------
    iou_thresh : float
        IOU overlap threshold for TP
    class_names : list of str
        optional, if provided, will print out AP for each class
    """
    def __init__(self, iou_thresh=0.5, class_names_mAP=[]):
        assert isinstance(class_names_mAP, (list, tuple))
        for name in class_names_mAP:
            assert isinstance(name, str), "must provide names as str"
        num = len(class_names_mAP)
        self.name = list(class_names_mAP) + ['mAP']
        self.num = len(self.name)
        self.reset()
        self.iou_thresh = iou_thresh
        self.class_names = class_names_mAP

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self._n_pos = defaultdict(int)
        self._score = defaultdict(list)
        self._match = defaultdict(list)

    def get(self):
        """Get the current evaluation result.
        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self._update()  # update metric at this time
        values = [x/y if y != 0 else float('nan') \
            for x, y in zip(self.sum_metric, self.num_inst)]


        return (self.name, values)

    # pylint: disable=arguments-differ, too-many-nested-blocks
    def update(self, pred_bboxes, pred_labels, pred_scores,
               gt_bboxes, gt_labels, gt_difficults=None):
        """Update internal buffer with latest prediction and gt pairs.
        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        gt_bboxes : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes with shape `B, M, 4`.
            Where B is the size of mini-batch, M is the number of ground-truths.
        gt_labels : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes labels with shape `B, M`.
        gt_difficults : mxnet.NDArray or numpy.ndarray, optional, default is None
            Ground-truth bounding boxes difficulty labels with shape `B, M`.
        """
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [x for x in a]
                try:
                    out = np.concatenate(out, axis=0)
                except ValueError:
                    out = np.array(out)
                return out
            return a

        if gt_difficults is None:
            gt_difficults = [None for _ in as_numpy(gt_labels)]

        if isinstance(gt_labels, list):
            if len(gt_difficults) != len(gt_labels) * gt_labels[0].shape[0]:
                gt_difficults = [None] * len(gt_labels) * gt_labels[0].shape[0]


        for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in zip(
                *[as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores,
                                        gt_bboxes, gt_labels, gt_difficults]]):
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]
            valid_gt = np.where(gt_label.flat >= 0)[0]
            gt_bbox = gt_bbox[valid_gt, :]
            gt_label = gt_label.flat[valid_gt].astype(int)
            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])
            else:
                gt_difficult = gt_difficult.flat[valid_gt]

            for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                self._n_pos[l] += np.logical_not(gt_difficult_l).sum()
                self._score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    self._match[l].extend((0,) * pred_bbox_l.shape[0])
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1

                iou = bbox_iou(pred_bbox_l, gt_bbox_l)
                gt_index = iou.argmax(axis=1)
                # set -1 if there is no matching ground truth
                gt_index[iou.max(axis=1) < self.iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            self._match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                self._match[l].append(1)
                            else:
                                self._match[l].append(0)
                        selec[gt_idx] = True
                    else:
                        self._match[l].append(0)

    def _update(self):
        """ update num_inst and sum_metric """
        aps = []
        recall, precs = self._recall_prec()
        for l, rec, prec in zip(range(len(precs)), recall, precs):
            ap = self._average_precision(rec, prec)
            print('ap', ap)
            print('l', l)
            # print('rec', rec)
            # print('prec', prec)
            aps.append(ap)
            if self.num is not None and l < (self.num - 1):
                self.sum_metric[l] = ap
                self.num_inst[l] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)

    def _recall_prec(self):
        """ get recall and precision from internal records """
        n_fg_class = max(self._n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for l in self._n_pos.keys():
            score_l = np.array(self._score[l])
            match_l = np.array(self._match[l], dtype=np.int32)

            order = score_l.argsort()[::-1]
            match_l = match_l[order]

            tp = np.cumsum(match_l == 1)
            fp = np.cumsum(match_l == 0)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            with np.errstate(divide='ignore', invalid='ignore'):
                prec[l] = tp / (fp + tp)
            # If n_pos[l] is 0, rec[l] is None.
            if self._n_pos[l] > 0:
                rec[l] = tp / self._n_pos[l]

        return rec, prec

    def _average_precision(self, rec, prec):
        """
        calculate average precision
        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        if rec is None or prec is None:
            return np.nan

        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], np.nan_to_num(prec), [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


class VOC07MApMetric(VOCMApMetric):
    """ Mean average precision metric for PASCAL V0C 07 dataset
    Parameters:
    ---------
    iou_thresh : float
        IOU overlap threshold for TP
    class_names : list of str
        optional, if provided, will print out AP for each class
    """
    def __init__(self, *args, **kwargs):
        #super(VOC07MApMetric, self).__init__(*args, **kwargs)
        VOCMApMetric.__init__(self,*args, **kwargs)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric
        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        #[print("%.3f %.3f"%(prec[i],rec[i])) for i in range(len(rec)) ]
        if rec is None or prec is None:
            return np.nan
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / 11.
        return ap

def bboxes_json2bboxes_gluon(bboxes_json):
    bboxes_gluon = []
    labels = []
    confidences = []
    for bbox_json in bboxes_json:
        top, left, height, width = bbox_json["objectPicY"], bbox_json["objectPicX"], bbox_json["objectHeight"], bbox_json["objectWidth"]
        #label = (bbox_json.get("objectTypes",["None"])[0])
        label = 0
        confidence = bbox_json.get("confidences",[0.])[0]/1e2
        bboxes_gluon.append([top,left,top+height,left+width])
        labels.append(label)
        confidences.append(confidence)
    if len(bboxes_gluon) == 0:
        bboxes_gluon = np.zeros((1,0,4))
        labels = np.zeros((1,0))
        confidences = np.zeros((1,0))
    else:
        bboxes_gluon = np.array([bboxes_gluon])
        labels = np.array([labels])
        confidences = np.array([confidences])
    return bboxes_gluon, labels, confidences


class MapTester:
    """
    label format is: [category, cx, cy, w, h] which (cx, cy, w, h) is normalized
    predict format is: [category, cx, cy, w, h, confidence] which (cx, cy, w, h) is normalized

    """
    def __init__(self, images_folder, labels_folder, predicts_folder, category_list):
        self.predicts_folder = predicts_folder
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.category_list = category_list

    def _load_img_size(self, path):
        img = cv2.imread(path)
        h, w, _ = img.shape
        return w, h


    def _get_label_box(self, line, img_size):
        """
        the output format is y1, x1, y2, x2
        :param line:
        :param img_size:
        :return:
        """
        img_w, img_h = img_size

        cx = float(line[1]) * img_w
        cy = float(line[2]) * img_h
        w = float(line[3]) * img_w
        h = float(line[4]) * img_h

        half_w = w / 2
        half_h = h / 2
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h

        return [int(y1), int(x1), int(y2), int(x2)]

    def _load_predict(self, path, img_size):
        bbox_list = []
        category_list = []
        confidence_list = []

        with open(path, 'r') as f:
            lines = f.readlines()
            # print(lines)
            if len(lines) == 0:
                # print('qqq')
                bboxes_gluon = np.zeros((1, 0, 4))
                labels = np.zeros((1, 0))
                confidences = np.zeros((1, 0))
                return bboxes_gluon, labels, confidences
            else:
                for line in lines:
                    line = line.rstrip('\n')
                    line = line.split()
                    category = int(line[0])
                    # bbox = self._get_box(line=line, img_size=img_size)
                    bbox = self._get_label_box(line=line, img_size=img_size)

                    confidence = float(line[5])
                    bbox_list.append(bbox)
                    category_list.append(category)
                    confidence_list.append(confidence)

                return np.array([bbox_list]), np.array([category_list]), np.array([confidence_list])

        # return pred_bboxes,pred_labels,pred_scores

    def _load_label(self, path, img_size):
        bbox_list = []
        category_list = []

        with open(path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                bboxes_gluon = np.zeros((1, 0, 4))
                labels = np.zeros((1, 0))
                confidences = np.zeros((1, 0))
                return bboxes_gluon, labels, confidences

            else:
                for line in lines:
                    line = line.rstrip('\n')
                    line = line.split()
                    category = int(line[0])
                    # bbox = self._get_box(line=line, img_size=img_size)
                    bbox = self._get_label_box(line=line, img_size=img_size)

                    bbox_list.append(bbox)
                    category_list.append(category)

                return np.array([bbox_list]), np.array([category_list])



    def run(self):
        # class_names_detection = ["LPR"]
        print('map running........')
        # voc07metric = VOC07MApMetric(class_names_mAP=["pillar", "buoy"])
        voc07metric = VOC07MApMetric(class_names_mAP=self.category_list)
        voc07metric.reset()
        for filename in tqdm(os.listdir(self.predicts_folder)):

            predict_path = '{}/{}'.format(self.predicts_folder, filename)
            # predict_path = '/media/taka/dataset/map_test/COCO_train2014_000000000009.txt'
            label_path = '{}/{}'.format(self.labels_folder, filename)
            image_path = '{}/{}jpg'.format(self.images_folder, filename[:-3])

            img_size = self._load_img_size(path=image_path)
            predict_bbox_list, predict_category_list, confidence_list = self._load_predict(
                path=predict_path, img_size=img_size)
            gt_box_list, gt_category_list = self._load_label(path=label_path, img_size=img_size)
            # print('predict_path', predict_path)
            # print('label_path', label_path)
            # print('img_path', image_path)

            # print('predict_bbox_list', predict_bbox_list)
            # print(predict_category_list.shape)
            # print(confidence_list.shape)
            # print('gt_box_list', gt_box_list.shape)
            # print('gt_category_list', gt_category_list.shape)

            voc07metric.update(pred_bboxes=predict_bbox_list,
                                   pred_labels=predict_category_list,
                                   pred_scores=confidence_list,
                                   gt_bboxes=gt_box_list,
                                   gt_labels=gt_category_list)
        results = voc07metric.get()
        print(results)
        for metric, mAP in zip(*results):
            print("%10s: %8.4f%%"%(metric,(mAP*1e2)))



def main():
    # root_folder = '/media/taka/dataset/map_test'
    category_list = ['pillar', 'buoy']
    # root_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far'
    root_folder = '/home/taka/river_data/cropped'
    predicts_folder = '{}/predicts'.format(root_folder)
    images_folder = '{}/images'.format(root_folder)
    labels_folder = '{}/labels'.format(root_folder)
    map_tester = MapTester(predicts_folder=predicts_folder, labels_folder=labels_folder,
                           images_folder=images_folder, category_list=category_list)
    map_tester.run()


if __name__ == "__main__":
    main()