
import os

class FolderToFile:
    def __init__(self):
        pass


    def run(self, folder_path, output_file):
        files = os.listdir(folder_path)
        with open(output_file, 'w') as f:
            for filename in files:
                print(filename)
                path = '{}/{}'.format(folder_path, filename)
                f.write(path + '\n')



def main():
    folder_path = '/media/taka/dataset/coco/labels/val2014'
    output_file = '/media/taka/dataset/coco/labels/val2014.txt'
    f2f = FolderToFile()
    f2f.run(folder_path=folder_path, output_file=output_file)

if __name__ == "__main__":
    main()
