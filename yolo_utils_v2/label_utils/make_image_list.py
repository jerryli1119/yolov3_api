
import os
import random


class DataSetFilter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.path_list = []
        self.category_dict = {
            '1': 'pillar',
            '0': 'buoy'
        }
        self.add_ratio = 0.25

    def _get_category(self, lines):
        for line in lines:
            line = line.rstrip('\n')
            line = line.split()
            category = line[0]
        return category

    def _add_to_list_randomly(self, path):
        print(random.random())
        # if random.random() < self.add_ratio:
        #     self.path_list.append(path)

    def _parse_label(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                category = self._get_category(lines=lines)
                if self.category_dict[category] == 'buoy':
                    # self._add_to_list_randomly(path=path)
                    self.path_list.append(path)
                else:
                    pass

    def run(self, label_folder):
        files = os.listdir(label_folder)
        for filename in files:
            label_path = '{}/{}'.format(label_folder, filename)
            print(label_path)
            self._parse_label(path=label_path)

        print(len(self.path_list))



def main():
    label_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/0/labels'
    output_path = ''
    dataset_filter = DataSetFilter(output_path=output_path)
    dataset_filter.run(label_folder=label_folder)

main()