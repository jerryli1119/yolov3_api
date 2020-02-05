import cv2
from .idatawriter import IDataWriter


class EmptyWriter(IDataWriter):
    def __init__(self, output_folder=None):
        pass

    def write_data(self, img=None, filename=None):
       pass
