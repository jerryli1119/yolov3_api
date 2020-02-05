"""
frame to video
"""
import cv2
from .idatawriter import IDataWriter

class VideoWriter(IDataWriter):
    def __init__(self, fps, output_path, img_size):
        self.fps = fps
        self.output_path = output_path
        self.img_size = img_size
        self._video_write_init()

    def _video_write_init(self):
        width, height = self.img_size
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path,
                              fourcc, self.fps, (int(width), int(height)))
        self.video_writer = out

    def write_data(self, img, filename=None):
        self.video_writer.write(img)

