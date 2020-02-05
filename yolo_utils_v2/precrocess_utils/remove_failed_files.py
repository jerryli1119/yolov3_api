import os

class RemoveFailedFiles:
    def __init__(self, failed_file_list, image_folder, label_folder):
        self.failed_file_list = failed_file_list
        self.image_folder = image_folder
        self.label_folder = label_folder

    def remove_failed_files(self):
        with open(self.failed_file_list, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                print(line)
                line = line.split('/')
                filename = line[-1]
                filename = filename[:-4]
                print(filename)
                image_path = '{}/{}.jpg'.format(self.image_folder, filename)
                label_path = '{}/{}.txt'.format(self.label_folder, filename)

                os.remove(image_path)
                os.remove(label_path)



def main():
    failed_file_list = '/media/taka/dataset/projects/yolo_utils/failed_files.txt'
    image_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/images'
    label_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/cropped/labels'

    rff = RemoveFailedFiles(failed_file_list=failed_file_list,
                            image_folder=image_folder, label_folder=label_folder)
    rff.remove_failed_files()

main()