def get_iou(boxA, boxB):
    # if boxes dont intersect
    if not _boxes_intersect(boxA, boxB):
        return 0
    inter_area = _get_intersection_area(boxA, boxB)
    union_area = _get_union_areas(boxA, boxB, inter_area=inter_area)
    # intersection over union
    iou = inter_area / union_area
    assert iou >= 0
    return iou


def _boxes_intersect(boxA, boxB):
    if boxA.x_min > boxB.x_max:
        return False  # boxA is right of boxB
    if boxB.x_min > boxA.x_max:
        return False  # boxA is left of boxB
    if boxA.y_max < boxB.y_min:
        return False  # boxA is above boxB
    if boxA.y_min > boxB.y_max:
        return False  # boxA is below boxB
    return True

def _get_intersection_area(boxA, boxB):
    xA = max(boxA.x_min, boxB.x_min)
    yA = max(boxA.y_min, boxB.y_min)
    xB = min(boxA.x_max, boxB.x_max)
    yB = min(boxA.y_max, boxB.y_max)
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def _get_union_areas(boxA, boxB, inter_area=None):
    area_A = _get_area(boxA)
    area_B = _get_area(boxB)
    if inter_area is None:
        inter_area = _get_intersection_area(boxA, boxB)
    return float(area_A + area_B - inter_area)

def _get_area(box):
    return (box.x_max - box.x_min + 1) * (box.y_max - box.y_min + 1)


def non_maximum_suppression(detections, threshold=.3):
    if not detections:
        return []
    detections = sorted(detections, key=lambda det: det.score, reverse=True)
    new_detections=[detections[0]]
    
    for detection in detections:
        for new_detection in new_detections:
            if get_iou(detection.box, new_detection.box) > threshold:
                break
        else:
            new_detections.append(detection)
    return new_detections