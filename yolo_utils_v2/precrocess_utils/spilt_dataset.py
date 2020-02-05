import random
import os

class DataSetSplitter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.save_path_dict = self._path_init()
        self.ratio = {
            'train': 0.8,
            'test': 0.1,
            'val': 0.1
        }

    # def _ration_init(self):
    #     train_test_interval = self.ratio['train']

    def _path_init(self):
        train_path = '{}/train.txt'.format(self.output_folder)
        test_path = '{}/test.txt'.format(self.output_folder)
        val_path = '{}/val.txt'.format(self.output_folder)
        save_path_dict = {
            'train': train_path,
            'test': test_path,
            'val': val_path
        }
        return save_path_dict

    def _save_to_file(self, path, line):
        with open(path, 'a') as f:
            f.write(line+ '\n')

    def _split_factory(self, random_num):
        if random_num < self.ratio['train']:
            save_path = self.save_path_dict['train']
        elif random_num >= self.ratio['train'] and random_num < self.ratio['train'] + self.ratio['test']:
            save_path = self.save_path_dict['test']
        else:
            save_path = self.save_path_dict['val']
        return save_path

    def make_file_list(self, folder):
        for filename in os.listdir(folder):
            img_path = '{}/{}'.format(folder, filename)
            print(img_path)
            output_file = '/media/taka/dataset/label_preprocess_data/frames/reiver_data/data_list/images_list.txt'
            self._save_to_file(path=output_file, line=img_path)





    def run(self, path):
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                print(line)
                random_num = random.random()
                print(random_num)
                save_path = self._split_factory(random_num=random_num)
                self._save_to_file(path=save_path, line=line)



def main():
    # file_list_path = '/media/taka/dataset/label_preprocess_data/frames/small_object_test/image_list.txt'
    # output_folder = '/media/taka/dataset/label_preprocess_data/frames/small_object_test/data_list'
    # img_folder = '/media/taka/dataset/label_preprocess_data/frames/reiver_data/images'
    file_list_path = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/image_list.txt'
    output_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/data_list'
    img_folder = '/media/taka/dataset/label_preprocess_data/frames/reiver_data/images'
    splitter = DataSetSplitter(output_folder=output_folder)
    splitter.run(path=file_list_path)
    # splitter.make_file_list(folder=img_folder)


main()