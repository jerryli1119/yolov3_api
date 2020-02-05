import os
import random


class ImageListHelper:
    def __init__(self, output_path, image_folder):
        self.output_path = output_path
        self.image_folder = image_folder

    def _save_to_files(self, img_path_list):
        with open(self.output_path, 'w') as f:
            for path in img_path_list:
                f.write(path + '\n')

    def make_list(self):
        img_path_list = []
        files = os.listdir(self.image_folder)
        for filename in files:
            img_path = '{}/{}'.format(self.image_folder, filename)
            # f.write(img_path + '\n')
            img_path_list.append(img_path)
        img_path_list = self._reorder(img_path_list=img_path_list)
        self._save_to_files(img_path_list=img_path_list)

    def _reorder(self, img_path_list):
        max_index = len(img_path_list) - 1
        reorder_times = int(0.5 * max_index)
        for i in range(reorder_times):
            start_index = random.randint(0, max_index)
            end_index = random.randint(0, max_index)
            buf = img_path_list[start_index]
            img_path_list[start_index] = img_path_list[end_index]
            img_path_list[end_index] = buf
        return img_path_list


class ListMerger:
    def __init__(self, img_lists, root_folder, output_filename):
        self.img_lists = img_lists
        self.root_folder = root_folder
        self.output_filename = output_filename
        self.res = []


    def _read_lines(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.res.append(line)

    def _write_file(self):
        output_path = '{}/{}'.format(self.root_folder, self.output_filename)
        with open(output_path, 'w') as f:
            for line in self.res:
                f.write(line)

    def run(self):
        for img_list in self.img_lists:
            path = '{}/{}'.format(self.root_folder, img_list)
            self._read_lines(path=path)

        self._write_file()





def main():
    # cfg = {
    #     'label_folder': '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped',
    #     'output_path': '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/image_list',
    #     '0': 1,
    #
    # }


    # output_path = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/image_list'
    # image_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/images'
    folder = '/media/taka/dataset/river_data/fake_pillar/cropped_images'
    output_path = '/media/taka/dataset/river_data/fake_buoy/fake_pillar_list.txt'


    img_list_helper = ImageListHelper(output_path=output_path, image_folder=folder)
    img_list_helper.make_list()

def main2():
    root_folder = '/media/taka/dataset/river_data/ture_plus_fake'
    img_lists = ['fake_buoy_list.txt', 'fake_pillar_list.txt', 'true_train.txt']
    output_filename = 'res.txt'
    xx = ListMerger(img_lists=img_lists, root_folder=root_folder, output_filename=output_filename)
    xx.run()

if __name__ == "__main__":
    main2()