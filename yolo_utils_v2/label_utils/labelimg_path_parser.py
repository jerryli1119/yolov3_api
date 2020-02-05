"""
<annotation>
	<folder>images</folder>
	<filename>00000.jpg</filename>
	<path>/media/taka/dataset/label_preprocess_data/frames/reiver_data/images/00000.jpg</path>
</annotation>
change the path in annotation for the different environment
"""
import xml.etree.ElementTree as ET
import os


class LabelimgPathParse:
    def __init__(self, xml_folder, output_folder, new_image_folder):
        self.xml_folder = xml_folder
        self.output_folder = output_folder
        self.new_image_folder = new_image_folder

    def _path_modification(self, filename):
        xml_path = '{}/{}'.format(self.xml_folder, filename)
        output_path = '{}/{}'.format(self.output_folder, filename)
        new_img_path = '{}/{}.jpg'.format(self.new_image_folder, filename[:-4])
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # print(root)
        img_path = root.find('path')
        img_path.text = new_img_path
        tree.write(output_path)

    def run(self):
        for filename in os.listdir(self.xml_folder):
            print(filename)
            self._path_modification(filename=filename)



def main():
    # xml_folder = '/media/taka/dataset/label_preprocess_data/frames/reiver_data/test_xml'
    # output_folder = '/media/taka/dataset/label_preprocess_data/frames/reiver_data/test_xml_output'
    # new_img_folder = 'xxx/xxx/xxx'

    xml_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/4/label'
    output_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/4/new_label'
    new_img_folder = '/media/taka/dataset/pillar_and_bouy/xx/finish_label/4/images'

    path_parser = LabelimgPathParse(xml_folder=xml_folder, output_folder=output_folder, new_image_folder=new_img_folder)
    path_parser.run()

if __name__ == '__main__':
    main()