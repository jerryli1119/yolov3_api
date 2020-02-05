import os


class LabelAnalysis:
    def __init__(self):
        self.count = 0

    def _load_label(self, path):
        with open(path, 'r') as f:
            # print(path)
            lines = f.readlines()
            # print(lines)
            if len(lines) == 0:
                print(path)
                self.count += 1

    def run(self, folder):

        files = os.listdir(folder)
        for filename in files:
            path = '{}/{}'.format(folder, filename)
            self._load_label(path=path)

        print(self.count)


def main():
    folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/predicts'
    la = LabelAnalysis()
    la.run(folder=folder)

if __name__ == '__main__':
    main()