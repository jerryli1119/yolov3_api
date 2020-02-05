
"""
give the crop size x, y, crop the image of the bounding box center
only for one object
"""
import cv2
import numpy as np
import os
import random
# from yolo_utils.format_transform import x1y1x2y2_to_cxcywh
from yolo_utils_v2.parser_helper import FilenameParser, LabelParser
from yolo_utils_v2.format_transform import x1y1x2y2_to_cxcywh_normalized
from yolo_utils_v2.box_drawer import BoxDrawer
from yolo_utils_v2.data_loader.image_loader import ImageLoader
from yolo_utils_v2.data_writer.image_writer import ImageWriter



"""
if the iou between crop image and target is smaller than the iou_threshold
select a new crop region
"""
class CropperV3:
    def __init__(self, crop_size, filename_parser,
                 image_output_folder, label_output_folder):

        """

        :param crop_size: (w, h)
        :param filename_parser:
        :param image_output_folder:
        :param label_output_folder:
        """

        self.crop_size_x, self.crop_size_y = crop_size
        # self.box_drawer = box_drawer
        # self.label_parser = label_parser
        self.filename_parser = filename_parser
        self.image_output_folder = image_output_folder
        self.label_output_folder = label_output_folder
        self.iou_threshold = 0.3

    def _load_label(self, path):
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                line = line.split()
        return line

    def _cxcywh_to_y1x1y2x2(self, box):
        """
        :param box: with (cx, cy, w, h) format
        :return: (top, left, bottom, right)
        """
        cx, cy, w, h = box
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        # return y1, x1, y2, x2

        return x1, y1, x2, y2

    def _get_iou(self, box1, box2):
        """
        the box format need to transfer to (top, left, bottom, right)
        :param box1: with (cx, cy, w, h) format
        :param box2: with (cx, cy, w, h) format
        :return: iou
        """
        # box1 = self._cxcywh_to_y1x1y2x2(box=box1)
        box2 = self._cxcywh_to_y1x1y2x2(box=box2)
        # print('box1', box1)
        # print('box2', box2)
        cx1, cy1, cx2, cy2 = box1
        gx1, gy1, gx2, gy2 = box2

        carea = (cx2 - cx1) * (cy2 - cy1)
        garea = (gx2 - gx1) * (gy2 - gy1)
        x1 = max(cx1, gx1)
        y1 = max(cy1, gy1)
        x2 = min(cx2, gx2)
        y2 = min(cy2, gy2)
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h  # the area fo C∩G
        iou = area / (carea + garea - area)
        return iou



    def _select_region(self, w, h, target_box_list, filename):
        """
        if the iou between target box and cropped box is zero, the target box is not inside the crop region
        select the cropped region randomly
        check the target is inside the cropped image
        if try too many times the target is still not inside the cropped image
        add the path to the failed_list and to be remove
        :param w: the image width
        :param h: the image height
        :param target_box_list: the target box list in this format
        [{'category': '16', 'box': (312, 302, 625, 572)}, {'category': '29', 'box': (413, 331, 355, 291)}]
        :param filename:
        :return: the cropped image boundary x1, y1, x2, y2
        """
        get_box = True
        count = 0
        while get_box:
            x1 = random.randint(0, w - self.crop_size_x)
            y1 = random.randint(0, h - self.crop_size_y)
            # x1 = 0
            # y1 = 0
            x2 = x1 + self.crop_size_x
            y2 = y1 + self.crop_size_y
            crop_box = x1, y1, x2, y2
            # print(crop_box)
            target_inside_list = []
            for target_box_info in target_box_list:
                target_box = target_box_info['box']
                iou = self._get_iou(box1=crop_box, box2=target_box)
                # print('iou', iou)
                if iou > self.iou_threshold:
                    # target_box = self._check_box_boundary()
                    # target_box_info['box'] = target_box
                    target_inside_list.append(target_box_info)
            if len(target_inside_list) > 0:
                break

            count += 1
            if count > 500:
                with open('failed_files.txt', 'a') as f:
                    print(filename)
                    f.write(filename + '\n')
                break
        return crop_box, target_inside_list




            # get_box = self._check_box_inside(box=box, crop_box=(x1, y1, x2, y2))



    def _load_img_shape(self, img):
        h, w, _ = img.shape
        return w, h

    def _parse_label(self, label, img_w, img_h):
        """
        from yolo label format to x1y1x2y2
        :param label:
        :param img_w:
        :param img_h:
        :return: (x1, y1, x2, y2)
        """
        category, cx, cy, w, h = label
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
        return res

    def _crop_img(self, img, crop_box):
        x1, y1, x2, y2 = crop_box
        w = x2 - x1
        h = y2 - y1
        img = img[y1: y1 + h, x1: x1 + w, :]
        return img

    def _check_box_boundary(self, box):
        """
        if out of boundary, fix the value
        :param box:
        :return:
        """
        x1, y1, x2, y2 = box
        if x1 < 0:
            x1 = 5
        if y1 < 0:
            y1 = 5
        if x2 > self.crop_size_x:
            x2 = self.crop_size_x - 5
        if y2 > self.crop_size_y:
            y2 = self.crop_size_y - 5
        return x1, y1, x2, y2


    def _make_new_label(self, inside_box_list, crop_box):
        new_label_list = []
        for box_info in inside_box_list:
            box = box_info['box']
            category = box_info['category']
            tx1, ty1, tx2, ty2 = box
            x1, y1, x2, y2 = crop_box
            new_x1 = tx1 - x1
            new_y1 = ty1 - y1
            new_x2 = tx2 - x1
            new_y2 = ty2 - y1
            box = (new_x1, new_y1, new_x2, new_y2)
            # print('xxx', box)
            box = self._check_box_boundary(box=box)
            # print('qqq', box)
            # new_x = new_x / self.crop_size_x
            # new_y = new_y / self.crop_size_y
            # w = w / self.crop_size_x
            # h = h / self.crop_size_y
            # # new_label = [category, new_x, new_y, w, h]
            # print('crop_size_x', self.crop_size_x)
            # print('crop_size_y', self.crop_size_y)
            new_label = {
                'category': category,
                'box': box
            }
            new_label_list.append(new_label)
        return new_label_list

    def _save_label(self, new_label_list, output_path):
        with open(output_path, 'w') as f:
            for label in new_label_list:
                box = label['box']
                # box = x1y1x2y2_to_cxcywh(box=box, img_w=self.crop_size_x, img_h=self.crop_size_y)
                box = x1y1x2y2_to_cxcywh_normalized(box=box, img_w=self.crop_size_x, img_h=self.crop_size_y)
                category = label['category']
                f.write(category + ' ')
                for i in range(len(box) - 1):
                    x = box[i]
                    f.write(str(x) + ' ')
                f.write(str(box[-1]) + '\n')

    def _save_img(self, img, output_path):
        cv2.imwrite(output_path, img)

    def _get_target_box_info_list(self, label_path, img_w, img_h):
        target_box_info_list = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                line = line.split()
                target_box_info = self._parse_label(label=line, img_w=img_w, img_h=img_h)
                target_box_info_list.append(target_box_info)
        return target_box_info_list



    def run(self, img_path, label_path, filename):
        img = cv2.imread(img_path)
        img_w, img_h = self._load_img_shape(img=img)
        target_box_info_list = self._get_target_box_info_list(label_path=label_path, img_w=img_w, img_h=img_h)
        # print('target_box_info_list', target_box_info_list)
        crop_box, target_inside_list = self._select_region(
            w=img_w, h=img_h, target_box_list=target_box_info_list, filename=filename)
        # # # print(box)
        # print(crop_box)
        print('target_inside_list', target_inside_list)
        new_label_list = self._make_new_label(inside_box_list=target_inside_list, crop_box=crop_box)
        # print('new_label_list', new_label_list)
        cropped_img = self._crop_img(img=img, crop_box=crop_box)
        filename = self.filename_parser.parse_filename(file_path=img_path)
        image_output_path = '{}/{}.jpg'.format(self.image_output_folder, filename)
        label_output_path = '{}/{}.txt'.format(self.label_output_folder, filename)
        self._save_img(img=cropped_img, output_path=image_output_path)
        self._save_label(new_label_list=new_label_list, output_path=label_output_path)

        # filename = '{}.jpg'.format(filename)
        # self.box_drawer.draw_box2(box_list=new_label_list, img=cropped_img, filename=filename)


def main():
    images_folder = '/media/taka/dataset/river_data/cropped/test/images'
    labels_folder = '/media/taka/dataset/river_data/cropped/test/labels'
    images_output_folder = '/media/taka/dataset/river_data/cropped/test/cropped_images'
    labels_output_folder = '/media/taka/dataset/river_data/cropped/test/cropped_labels'
    filename_parser = FilenameParser()

    cropper = CropperV3(crop_size=(380, 500),
                        filename_parser=filename_parser, image_output_folder=images_output_folder,
                        label_output_folder=labels_output_folder)
    for file in os.listdir(images_folder):
        filename = filename_parser.parse_filename(file)
        print(filename)
        img_path = '{}/{}.jpg'.format(images_folder, filename)
        label_path = '{}/{}.txt'.format(labels_folder, filename)
        cropper.run(img_path=img_path, label_path=label_path, filename=filename)


def draw_box():
    images_folder = '/media/taka/dataset/river_data/cropped/test/cropped_images'
    labels_folder = '/media/taka/dataset/river_data/cropped/test/cropped_labels'
    output_folder = '/media/taka/dataset/river_data/cropped/test/output'

    filename_parser = FilenameParser()

    box_drawer = BoxDrawer()
    label_parser = LabelParser()
    data_writer = ImageWriter(output_folder=output_folder)
    for file in os.listdir(images_folder):
        filename = filename_parser.parse_filename(file)
        img_path = '{}/{}.jpg'.format(images_folder, filename)
        label_path = '{}/{}.txt'.format(labels_folder, filename)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        labels = label_parser.parse_label(label_path=label_path, img_w=w, img_h=h)
        print(labels)
        frame = box_drawer.draw_box_with_file(box_list=labels, img=img, filename=filename)
        data_writer.write_data(img=frame, filename=filename)



if __name__ == "__main__":
    pass
    # main()
    # draw_box()