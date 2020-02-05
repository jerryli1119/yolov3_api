import cv2


class ImageResizer:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        # print('resize', img.shape)
        return img


class EmptyResizer:
    def __init__(self):
        pass

    def __call__(self, img):
        # print('xxx')
        return img

