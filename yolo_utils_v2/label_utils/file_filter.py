
import os


class FileFilter:
    """
    remove the specific object in the dataset
    """
    def __init__(self, remove_id, label_folder, image_folder):
        self.remove_id = remove_id
        self.label_folder = label_folder
        self.image_folder = image_folder

    def _remove_file(self, filename):
        label_path = '{}/{}.txt'.format(self.label_folder, filename)
        image_path = '{}/{}.jpg'.format(self.image_folder, filename)
        os.remove(image_path)
        os.remove(label_path)

    def _get_name_id(self, line):
        line = line.rstrip('\n')
        line = line.split()
        name_id = line[0]
        return name_id

    def _check_file(self, filename):
        filename = filename[:-4]
        # print(filename)
        label_path = '{}/{}.txt'.format(self.label_folder, filename)
        with open(label_path, 'r') as f:
            for line in f.readlines():
                name_id = self._get_name_id(line=line)
                if name_id == self.remove_id:
                    print(filename)
                    self._remove_file(filename=filename)
    def run(self):
        for filename in os.listdir(self.label_folder):
            self._check_file(filename=filename)




def main():
    remove_id = '80'
    label_folder = '/media/taka/dataset/coco/labels/val2014'
    image_folder = '/media/taka/dataset/coco/images/val2014'
    ff = FileFilter(remove_id=remove_id, label_folder=label_folder, image_folder=image_folder)
    ff.run()

if __name__ == "__main__":
    main()