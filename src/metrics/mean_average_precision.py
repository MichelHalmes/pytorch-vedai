import sys 
from collections import Counter

import numpy as np

from metrics.intersection_over_union import get_iou


def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def GetPascalVOCMetrics(ground_truths,
                        detections,
                        IOUThreshold=0.005): # TODO: IOUThreshold=.5
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
        dict['class']: class representing the current dictionary;
        dict['precision']: array with the precision values;
        dict['recall']: array with the recall values;
        dict['AP']: average precision;
        dict['interpolated precision']: interpolated precision values;
        dict['interpolated recall']: interpolated recall values;
        dict['total positives']: total number of ground truth positives;
        dict['total TP']: total number of True Positive detections;
        dict['total FP']: total number of False Negative detections;
    """
    ret = []  # list containing metrics (precision, recall, average precision) of each class
    # Get all classes
    classes = {gt[1] for gt in ground_truths}
    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if d[1] == c]
        # Get only ground truths of class c
        gts = []
        [gts.append(g) for g in ground_truths if g[1] == c]
        npos = len(gts)
        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of gts for each image
        det = Counter([cc[0] for cc in gts])
        for key, val in det.items():
            det[key] = np.zeros(val)
        # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))
            # Find ground truth image
            gt = [gt for gt in gts if gt[0] == dects[d][0]]
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                # print('Ground truth gt => %s' % (gt[j][3],))
                iou = get_iou(dects[d][3], gt[j][3])
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                # print("FP")
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        [ap, mpre, mrec, _] = CalculateAveragePrecision(rec, prec)
        # add class result in the dictionary to be returned
        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }
        ret.append(r)
    return ret




def get_mean_average_precision(ground_truths, detections):
    acc_AP = 0
    validClasses = 0
    ground_truths_merged = [gt for batch_gt in ground_truths for gt in batch_gt ]
    detections_merged = [det for batch_det in detections for det in batch_det ]
    results = GetPascalVOCMetrics(ground_truths_merged, detections_merged)

    for metricsPerClass in results:
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap

    mAP = acc_AP / validClasses
    return mAP