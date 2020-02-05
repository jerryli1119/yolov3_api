

def cxcywh_to_x1y1x2y2_normalized(box, img_w, img_h):
    """
    :param box: the yolo label format (cx, cy, w, h) which are normalized
    :param img_w:
    :param img_h:
    :return: (left, top, right, bottom)
    """
    cx, cy, w, h = box
    cx = (float(cx) * img_w)
    cy = (float(cy) * img_h)
    w = (float(w) * img_w)
    h = (float(h) * img_h)
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2


def cxcywh_to_x1y1x2y2(box):
    """
    :param box: the yolo label format (cx, cy, w, h) which are normalized
    :param img_w:
    :param img_h:
    :return: (left, top, right, bottom)
    """
    cx, cy, w, h = box
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2

def x1y1x2y2_to_cxcywh_normalized(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    cx = cx / img_w
    cy = cy / img_h
    w = w / img_w
    h = h / img_h
    return cx, cy, w, h

def x1y1x2y2_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h

