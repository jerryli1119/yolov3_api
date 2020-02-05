

class FilenameParser:

    def parse_filename(self, file_path):
        filename = file_path.split('/')
        filename = filename[-1]
        return filename[:-4]


class LabelParser:
    """
    from yolo label format to x1y1x2y2
    accept the label or predict file
    if the label file is empty, the bbox set to (0, 0, 0, 0)

    """
    def __init__(self):
        pass

    def parse_label(self, label_path, img_w, img_h):
        """
        :param label:
        :param img_w:
        :param img_h:
        :return:
        res = {'category': category,
                'box': (int(x1), int(y1), int(x2), int(y2))
                }
        """
        label_list = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                line = line.split()
                # print(line)
                if len(line) == 5:
                    category, cx, cy, w, h = line
                else:
                    category, cx, cy, w, h, _predict = line
                cx = float(cx) * img_w
                cy = float(cy) * img_h
                w = float(w) * img_w
                h = float(h) * img_h
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                res = {
                    'category': category,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                }
                label_list.append(res)
        if len(label_list) == 0:
            res = {
                'category': 'no',
                'box': (0, 0, 0, 0)
            }
            label_list.append(res)
        return label_list
