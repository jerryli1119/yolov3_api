
import cv2
import numpy as np
from .idataloader import IDataLoader
from .image_resizer import ImageResizer, EmptyResizer



class VideoLoader(IDataLoader):
    def __init__(self, file_path, format_transfer, resize=None, frame_interval=1):
        self.file_path = file_path
        self._resizer = self._resizer_init(resize=resize)
        self.format_transfer = format_transfer
        self.frame_interval = frame_interval

    def _resizer_init(self, resize):
        if resize is None:
            resizer = EmptyResizer()
        else:
            resizer = ImageResizer(size=resize)
            
        return resizer


    def frame_iter(self):
        count = 0
        vidcap = cv2.VideoCapture(self.file_path)

        "skip the videp"
        #for _ in range(13000):
        #    success, image = vidcap.read()

        success, image = vidcap.read()
        img_shape = image.shape

        while success:
            success, image = vidcap.read()
            filename = str(count).zfill(5)

            if success:
                img = image
                # yield image, filename
            else:
                img = np.zeros(img_shape, dtype=np.uint8)
                # yield np.zeros(img_shape, dtype=np.uint8), filename
            img = self._resizer(img)
            img = self.format_transfer(img)
            if count % self.frame_interval == 0:
                yield img, filename
            count += 1

    def pre_load(self, max_num=10000000):
        data_list = []
        count = 0
        vidcap = cv2.VideoCapture(self.file_path)
        success, image = vidcap.read()

        while success:
            success, image = vidcap.read()
            image = self._resizer(image)
            image = self.format_transfer(image)


            filename = str(count).zfill(5)
            if count % self.frame_interval == 0:
                data_list.append((image, filename))
            count += 1
            if count > max_num:
                break
            
        return data_list




def test():
    file_path = '/media/taka/dataset/test_video/V_20180603_142437_vHDR_On.mp4'
    frame_iter = VideoLoader(file_path=file_path).frame_iter()
    for frame in frame_iter:
        print(frame.shape)


if __name__ == "__main__":
    test()

