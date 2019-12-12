import sys 
from collections import Counter

import numpy as np

from evaluate.intersection_over_union import get_iou


def calculate_average_precision(recall, precision):
    mrec = [0] + list(recall) + [1]
    mpre = [0] + list(precision) + [0]

    # We use the right cummulative maximum for the precision
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    # We integrate the area under the precision-recall curve
    ap = 0
    for i in range(len(mrec)-1):
        ap += (mrec[i+1] - mrec[i]) * mpre[i+1]

    return ap, mpre[:-1], mrec[:-1]


def get_pascal_voc_metrics(ground_truths,
                        detections,
                        IOUThreshold=0.5):
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Get
    Args:
        ground_truths: List with all ground truths [(imageName,class,confidence=1, (bb coordinates XYX2Y2))]
        detections: List with all detections [(imageName,class,confidence,(bb coordinates XYX2Y2))])
    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
        IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
        (default value = 0.5);
    Returns:
        A list of dictionaries. Each dictionary contains information and metrics of each class.
        The keys of each dictionary are:
         * class: class representing the current dictionary;
         * precision: array with the precision values;
         * recall: array with the recall values;
         * AP: average precision;
         * interpolated precision: interpolated precision values;
         * interpolated recall: interpolated recall values;
         * total positives: total number of ground truth positives;
         * total TP: total number of True Positive detections;
         * total FP: total number of False Negative detections;
    """
    metrics = []  # list containing metrics (precision, recall, average precision) of each class
    # Get all classes
    classes = {gt.label for gt in ground_truths}
    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for class_label in classes:
        # Get only detection of class class_label
        class_dects = [d for d in detections if d.label == class_label]
        # Get only ground truths of class class_label
        class_gts = [g for g in ground_truths if g.label == class_label]
        TP = np.zeros(len(class_dects))
        FP = np.zeros(len(class_dects))
        # create dictionary with amount of class_gts for each image
        gts_cnt_per_image = Counter(gt.image_id for gt in class_gts)
        detected_gts_per_image = {image_id: np.zeros(gt_cnt) for image_id, gt_cnt in gts_cnt_per_image.items()}
        # sort detections by decreasing confidence
        class_dects = sorted(class_dects, key=lambda d: d.score, reverse=True)
        # Loop through detections
        for d_idx, dect in enumerate(class_dects):
            # Find ground truth image
            gts_in_image = [gt for gt in class_gts if gt.image_id == dect.image_id]
            iouMax = sys.float_info.min
            for gt_idx, gt in enumerate(gts_in_image):
                iou = get_iou(dect.box, gt.box)
                if iou > iouMax:
                    iouMax = iou
                    match_gt_idx = gt_idx
            # Assign detection as true positive/don't care/false positive
            if iouMax >= IOUThreshold:
                if detected_gts_per_image[dect.image_id][match_gt_idx] == 0:
                    TP[d_idx] = 1  # count as true positive
                    detected_gts_per_image[dect.image_id][match_gt_idx] = 1  # flag as already 'seen'
                else:
                    FP[d_idx] = 1  # count as false positive
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d_idx] = 1  # count as false positive
        # compute precision, recall and average precision
        npos = len(class_gts)
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        recall = acc_TP / npos
        precision = np.divide(acc_TP, (acc_FP + acc_TP))
        ap, mpre, mrec = calculate_average_precision(recall, precision)
        # add class result in the dictionary to be returned
        class_metrics_dict = {
            "class": class_label,
            "precision": precision,
            "recall": recall,
            "AP": ap,
            "interpolated precision": mpre,
            "interpolated recall": mrec,
            "total positives": npos,
            "total TP": np.sum(TP),
            "total FP": np.sum(FP)
        }
        metrics.append(class_metrics_dict)
    return metrics




def get_mean_average_precision(ground_truths, detections):
    ground_truths_merged = [gt for batch_gt in ground_truths for gt in batch_gt ]
    detections_merged = [det for batch_det in detections for det in batch_det ]
    metrics = get_pascal_voc_metrics(ground_truths_merged, detections_merged)

    sum_ap = 0
    valid_classes_cnt = 0
    for class_metrics_dict in metrics:
        cl = class_metrics_dict["class"]
        ap = class_metrics_dict["AP"]
        precision = class_metrics_dict["precision"]
        recall = class_metrics_dict["recall"]
        total_positives = class_metrics_dict["total positives"]
        total_TP = class_metrics_dict["total TP"]
        total_FP = class_metrics_dict["total FP"]

        if total_positives > 0:
            valid_classes_cnt = valid_classes_cnt + 1
            sum_ap += ap

    mAP = sum_ap / valid_classes_cnt
    return mAP