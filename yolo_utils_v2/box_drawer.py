# -*- coding: UTF-8 -*-
import abc
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import numpy

class IBoxDrawer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def draw_box_with_path(self):
        return NotImplementedError

    @abc.abstractmethod
    def draw_box_with_file(self):
        return NotImplementedError


class BoxDrawer(IBoxDrawer):
    """
    the box drawer takes the (x1, y1, x2, y2) format
    """
    def __init__(self, color=None):
        # self.output_folder = output_folder
        # self.color_setting = color
        # self.color_list = self._check_color(color=color)
        self.color = self._color_init(color=color)

    def _parse_filename(self, img_path):
        filename = img_path.split('/')
        filename = filename[-1]
        return filename

    def _color_init(self, color):
        if color is None:
            color = [random.randint(0, 255) for _ in range(3)]
        return color

    def _check_color(self, color):
        if color is None:
            color = self.color
        return color

    def _draw_one_box(self, box_info, img, color=None):
        tl = round(0.002 * max(img.shape[0:2])) + 1  # line thickness

        color = self._check_color(color=color)
        # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        # print(c1, c2)

        x1, y1, x2, y2 = box_info['box']
        labels = box_info['category']

        color = [0, 0, 255]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=tl)
        if labels:
            tf = max(tl - 1, 1)  # font thickness
            # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            # c2 = x1 + t_size[0], y1 - t_size[1] - 3
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)  # filled

            #cv2.putText(image, context, set, Font, size, color, line size, line style)
            cv2.putText(img, labels, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def draw_box_with_path(self, box_list, img_path, color=None):
        img = cv2.imread(img_path)
        # filename = self._parse_filename(img_path=img_path)
        for box_info in box_list:
            self._draw_one_box(box_info=box_info, img=img, color=color)

        # output_path = '{}/{}'.format(self.output_folder, filename)
        # cv2.imwrite(output_path, img)
        return img

    def draw_box_with_file(self, box_list, img, color=None):
        # img = cv2.imread(img_path)
        for box_info in box_list:
            self._draw_one_box(box_info=box_info, img=img, color=color)

        # output_path = '{}/{}'.format(self.output_folder, filename)
        # cv2.imwrite(output_path, img)
        return img


class EmptyBoxDrawer(IBoxDrawer):
    """
    the box drawer takes the (x1, y1, x2, y2) format
    """
    def __init__(self, color=None):
        # self.output_folder = output_folder
        self.color_setting = color
        # self.color_list = self._check_color(color=color)

    def draw_box_with_path(self, box_list, img_path):
        img = cv2.imread(img_path)
        return img

    def draw_box_with_file(self, box_list, img):
        return img


def main():
    pass


if __name__ == "__main__":
    main()