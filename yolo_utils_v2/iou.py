import numpy as np


def get_IOU(bbox1 ,bbox2):
    """
    Computes IOU between two bboxes in the form [left,top,right,bottom]
    """
    left_overlap   = np.maximum(bbox1[0], bbox2[0])
    top_overlap    = np.maximum(bbox1[1], bbox2[1])
    right_overlap  = np.minimum(bbox1[2], bbox2[2])
    bottom_overlap = np.minimum(bbox1[3], bbox2[3])
    width_overlap  = right_overlap -left_overlap
    height_overlap = bottom_overlap -top_overlap
    if width_overlap <0 or height_overlap <0:
        return 0.
    area_overlap = width_overlap *height_overlap
    area1 = (bbox1[2 ] -bbox1[0] ) *(bbox1[3 ] -bbox1[1])
    area2 = (bbox2[2 ] -bbox2[0] ) *(bbox2[3 ] -bbox2[1])
    IOU = float(area_overlap ) /(area1 +area2 -area_overlap)
    return IOU