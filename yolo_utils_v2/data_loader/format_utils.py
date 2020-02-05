from .idataloader import IFormatLoader
import cv2
import io
import numpy as np


class EmptyFormat:
    """
    just return the image
    """
    def __init__(self):
        pass

    @staticmethod
    def __call__(img):
        return img

class ImgToByte:
    """
    numpy array to byte string
    """
    def __init__(self):
        pass

    @staticmethod
    def __call__(img):
        img = cv2.imencode('.jpg', img)[1].tostring()
        return img



class ByteToImg:
    """
    byte string to numpy array
    """
    def __init__(self):
        pass

    @staticmethod
    def __call__(bstring):
        try:
            stream = io.BytesIO(bstring)
            # print(stream)
            # qq = stream.getvalue()
            img_array = np.fromstring(stream.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, 1)
            return img
            
        except Exception:
            raise IOError


