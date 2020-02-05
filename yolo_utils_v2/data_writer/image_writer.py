import cv2
from .idatawriter import IDataWriter


class ImageWriter(IDataWriter):
    def __init__(self, output_folder):
        self.output_folder = output_folder


    # def _parse_filename(self, img_path):
    #     filename = img_path.split('/')
    #     filename = filename[-1]
    #     return filename

    def write_data(self, img, filename):
        # filename = self._parse_filename(img_path=img_path)
        output_path = '{}/{}.jpg'.format(self.output_folder, filename)
        cv2.imwrite(output_path, img)
