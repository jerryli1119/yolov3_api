"""
change the category in the label file
step1: set the category_dict
step2: set the input and output folder
"""

import os

class ReCategory:
    def __init__(self, label_folder, output_folder, category_dict):
        self.label_folder = label_folder
        self.output_folder = output_folder
        self.category_dict = category_dict

    def _make_new_label(self, new_category, line):
        label = '{} {} {} {} {}'.format(new_category, line[1], line[2], line[3], line[4])
        return label

    def _change_category(self, filename):
        label_path = '{}/{}'.format(self.label_folder, filename)
        with open(label_path, 'r') as f:
            label_list = []
            for line in f.readlines():
                line = line.rstrip('\n')
                line = line.split()
                print(line)
                new_category = self.category_dict[line[0]]
                print(new_category)
                label_list.append(self._make_new_label(new_category=new_category, line=line))
        return label_list

    def _save_file(self, label_list, output_path):
        with open(output_path, 'w') as f:
            for label in label_list:
                f.write(label + '\n')

    def run(self):
        files = os.listdir(self.label_folder)
        for filename in files:
            print(filename)
            label_list = self._change_category(filename=filename)
            output_path = '{}/{}'.format(self.output_folder, filename)
            self._save_file(label_list=label_list, output_path=output_path)


def main():
    category_dict = {
        # '15': '0',
        # '16': '0',
        '0': '1',

    }
    # labels_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/9/labels'
    # output_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/9/new_labels'

    labels_folder = '/media/taka/dataset/river_data/fake_pillar/labels'
    output_folder = '/media/taka/dataset/river_data/fake_pillar/new_labels'
    re_cate = ReCategory(label_folder=labels_folder, output_folder=output_folder, category_dict=category_dict)
    re_cate.run()

main()