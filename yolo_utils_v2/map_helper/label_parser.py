import os


class LabelParser:
    """
    input:
    result format of yolov3 pytorch:
    [{'coordinate': [577, 523, 591, 554], 'category': 'XXX', 'confidence': 0.2739827334880829}]

    output:
    the yolo format cx, cy, w, h and confidence, which cx, cy, w, h are normalized

    """
    def __init__(self, coco_names, predicts_folder):
        """
        :param coco_names: coco.names file path
        :param predicts_folder: the output folder
        """
        # self.coco_names = '/media/taka/dataset/projects/yolov3_pytorch/yolov3/data/coco.names'
        self.coco_names = coco_names
        self.name_dict = self._make_category_dict(path=self.coco_names)
        self.predicts_folder = predicts_folder

    def _make_category_dict(self, path):
        name_dict = {}
        with open(path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.rstrip('\n')
                print(line)
                name_dict[line] = str(i)
        return name_dict




    def _bbox_parser(self, bbox, img_w, img_h):
        """
        x1y1x2y2 to cx, cy, w, h and normalized (yolo format)
        :param bbox:
        :return:
        """
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        w = (w / img_w)
        h = (h / img_h)
        cx = (cx / img_w)
        cy = (cy /img_h)
        return cx, cy, w, h

    def _check_file_exits(self, path):
        if os.path.exists(path):
            os.remove(path)

    def _save_empty(self, filename):
        # path = '/media/taka/dataset/label_preprocess_data/frames/' \
        #        'small_object_test/real_test_image/predict/{}.txt'.format(filename)
        path = '{}/{}'.format(self.predicts_folder, filename)
        self._check_file_exits(path=path)

        with open(path, 'w') as f:
            pass

    def _save(self, bbox, filename, confidence, category):
        # path = '/media/taka/dataset/label_preprocess_data/frames/' \
        #        'small_object_test/real_test_image/predict/{}.txt'.format(filename)
        path = '{}/{}'.format(self.predicts_folder, filename)
        # self._check_file_exits(path=path)
        with open(path, 'a') as f:
            f.write(category + ' ')
            for x in bbox:
                f.write(str(x) + ' ')
            f.write(str(confidence))
            f.write('\n')

            # for x in bbox[:3]:
            #     f.write(str(x) + ' ')
            # f.write(str(bbox[-1]))
            # f.write('\n')


    def __call__(self, res, filename, img):
        h, w, _ = img.shape
        filename = '{}.txt'.format(filename)

        if res is not None:
            for i, data in enumerate(res):
                # print(test_data)
                bbox = data['box']
                confidence = data['confidence']
                category = data['category']
                category = self.name_dict[category]
                bbox = self._bbox_parser(bbox=bbox, img_w=w, img_h=h)
                self._save(bbox=bbox, filename=filename, confidence=confidence, category=category)
        else:
            self._save_empty(filename=filename)


class EmptyParser:
    """
    the empty parser
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass



