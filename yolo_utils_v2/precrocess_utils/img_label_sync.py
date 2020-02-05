"""
make sure the number of images and number of labels is same
"""

import os


class ImageLabelSync:
    def __init__(self, folder):
        self.folder = folder
        self.images_folder = '{}/images'.format(folder)
        self.labels_folder = '{}/labels'.format(folder)
        self.image_files = os.listdir(self.images_folder)
        self.label_files = os.listdir(self.labels_folder)

    def _remove_images(self):
        files = os.listdir(self.images_folder)
        label_files = os.listdir(self.labels_folder)
        for filename in files:
            # print(filename)
            label_name = '{}.txt'.format(filename[:-4])
            # print(filename)
            if label_name not in label_files:
                img_path = '{}/{}'.format(self.images_folder, filename)
                print(img_path)
                os.remove(img_path)

    def _remove_labels(self):
        img_files = os.listdir(self.images_folder)
        label_files = os.listdir(self.labels_folder)
        for filename in label_files:
            # print(filename)
            img_name = '{}.jpg'.format(filename[:-4])
            # print(filename)
            if img_name not in img_files:
                label_path = '{}/{}'.format(self.labels_folder, filename)
                print(label_path)
                os.remove(label_path)

    def run(self):
        if len(self.image_files) > len(self.label_files):
            self._remove_images()
        else:
            self._remove_labels()


def main():
    # for i in range(0, 9):
    #     folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/{}'.format(str(i))
    #     img_label_sync = ImageLabelSync(folder=folder)
    #     img_label_sync.run()
    folder = '/media/taka/dataset/coco/some-coco'
    img_label_sync = ImageLabelSync(folder=folder)
    img_label_sync.run()

main()