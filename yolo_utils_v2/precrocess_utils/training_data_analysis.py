"""

"""
import os

class TrainingDataAnalysis:
    def __init__(self):
        self.category_count_dict = {}

    def _count_category(self, category):
        if category in self.category_count_dict:
            self.category_count_dict[category] += 1
        else:
            self.category_count_dict[category] = 0

    def _parse_label(self, path):
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                line = line.split()
                print(line)
                self._count_category(category=line[0])

    def run(self, label_folder):
        files = os.listdir(label_folder)
        for filename in files:
            print(filename)
            label_path = '{}/{}'.format(label_folder, filename)
            self._parse_label(path=label_path)
        print(self.category_count_dict)


def main():
    label_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/labels'
    tda = TrainingDataAnalysis()
    tda.run(label_folder=label_folder)


    # folder = label_folder.format(str(i))

    # for i in range(10):
    #
    #     folder = label_folder.format(str(i))
    #     tda.run(label_folder=folder)
    #     print(i)

main()