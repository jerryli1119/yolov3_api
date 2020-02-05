from .idataloader import IDataLoader, IFormatLoader
import cv2
import os
from .image_resizer import ImageResizer, EmptyResizer


class ImageLoader(IDataLoader):
    def __init__(self, file_path, format_transfer, resize=None):
        """
        :param file_path: the image list file path
        """
        self.file_path = file_path
        self._resizer = self._resizer_init(resize=resize)
        self.format_transfer = format_transfer

    def _resizer_init(self, resize):
        if resize is None:
            resizer = EmptyResizer()
        else:
            resizer = ImageResizer(size=resize)

        return resizer


    def _parse_filename(self, file_path):
        filename = file_path.split('/')
        filename = filename[-1]
        return filename[:-4]

    def frame_iter(self):
        """
        return a frame iterator
        :return: (image, filename)
        """
        with open(self.file_path) as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                img = cv2.imread(line)
                img = self._resizer(img)
                img = self.format_transfer(img)
                # img = cv2.imread(line)
                filename = self._parse_filename(file_path=line)
                yield img, filename



class ImageLoaderFromFolder(IDataLoader):
    def __init__(self, folder, format_transfer, resize=None):
        """
        :param folder: the images folder
        """
        self.folder = folder
        self._resizer = self._resizer_init(resize=resize)
        self.format_transfer = format_transfer


    def _resizer_init(self, resize):
        if resize is None:
            resizer = EmptyResizer()
        else:
            resizer = ImageResizer(size=resize)

        return resizer

    def _parse_filename(self, file_path):
        filename = file_path.split('/')
        filename = filename[-1]
        return filename[:-4]

    def frame_iter(self):
        files = os.listdir(self.folder)
        for filename in sorted(files):
            img_path = '{}/{}'.format(self.folder, filename)
            img = cv2.imread(img_path)
            img = self._resizer(img)
            img = self.format_transfer(img)
            filename = self._parse_filename(file_path=filename)
            yield img, filename

    def pre_load(self, max_num=10000000, mode=None):
        data_list = []
        files = os.listdir(self.folder)
        count = 0

        def _rb_image(path):
            with open(path, 'rb') as f:
                data = f.read()
            return data

        for filename in sorted(files):
            img_path = '{}/{}'.format(self.folder, filename)

            img = cv2.imread(img_path)
            img = self._resizer(img)
            img = self.format_transfer(img)
            # print('resize', img.shape)
            # if mode == 'rb':
            #     img = cv2.imencode('.jpg', img)[1].tostring()
            # if mode == 'rb':
            #     img = _rb_image(img_path)
            # else:
            #     img = cv2.imread(img_path)
            #     img = self.resizer(img)

            filename = self._parse_filename(file_path=filename)
            data_list.append((img, filename))
            count += 1
            if count > max_num:
                break

        return data_list


def test():
    path = '/home/taka/river_data/cropped/data_list/test.txt'
    data_loader = ImageLoader(path)
    frame_iter = data_loader.frame_iter()
    for f in frame_iter:
        print(f)


if __name__ == "__main__":
    test()