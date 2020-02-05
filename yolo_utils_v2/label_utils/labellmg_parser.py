


# from lxml import etree
import xml.etree.ElementTree as ET
import os
import re



def print_log(log_dict):
    for log in log_dict.items():
        print(log)


class CategroyMap:
    def __init__(self):
        self.mapping = {}
        self.mapping['buoy'] = '1'
        self.mapping['pillar'] = '0'
        # self.mapping['pile-lighthouse'] = '1'
        # self.mapping['person'] = '0'
        # self.mapping['boat'] = '8'
        # self.mapping['person'] = '0'
        # self.mapping['bicycle'] = '1'
        # self.mapping['car'] = '2'
        # self.mapping['motorbike'] = '3'
        # self.mapping['aeroplane'] = '4'
        # self.mapping['bus'] = '5'
        # self.mapping['train'] = '6'
        # self.mapping['truck'] = '7'
        # self.mapping['boat'] = '8'


class CategroyCounter:
    def __init__(self):
        self.counter = {}
        self.counter['lighthouse'] = 0
        self.counter['pile-lighthouse'] = 0
        self.counter['pillar'] = 0
        self.counter['buoy'] = 0


class Keys:
    path = 'path'
    size = 'size'
    width = 'width'
    height = 'height'
    bndbox = 'bndbox'
    object = 'object'
    xmin = 'xmin'
    ymin = 'ymin'
    xmax = 'xmax'
    ymax = 'ymax'
    name = 'name'


class YOLOFormat:
    def __init__(self):
        self.image_path = None
        self.obj_info_list = []
        # self.coordinate_list = []
        # self.x = None
        # self.y = None
        # self.width = None
        # self.height = None

    def set_image_path(self, path):
        self.image_path = path

    def add_obj_info(self, obj):
        self.obj_info_list.append(obj)


class ObjectInfo:
    def __init__(self):
        self.categroy_id = None
        # self.xmin = None
        # self.ymin = None
        # self.xmax = None
        # self.ymax = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None


class Xml2YoloParser:
    """
    from labelimg format to yolo fromat(class, cx, cy, w, h)
    example:
    annotation_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/_labels'
    output_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/labels'
    categroy_dict = {
        'pillar': '0',
        'buoy': '1'
    }
    xml_parser = Xml2YoloParser(
        categroy_dict=categroy_dict, annotation_folder=annotation_folder,
        output_folder=output_folder)
    xml_parser.run()
    """
    def __init__(self, categroy_dict, annotation_folder, output_folder):
        """
        :param categroy_dict: the category mapping dict, example:
        category_dict = {
        'pillar': '0',
        'buoy': '1'
        }
        :param annotation_folder:
        :param output_folder:
        """
        self.annotation_folder = annotation_folder
        self.output_folder = output_folder

        self.root = None
        self.yoloformat_list = []
        self.categroy_dict = categroy_dict
        # self.categroy_counter = categroy_counter

        self.regular_pattern_init()


    def regular_pattern_init(self):
        self.pattern = re.compile(r'\d{1,5}')

    def get_root(self, xml_path):
        tree = ET.parse(xml_path)
        self.root = tree.getroot()
        # print(self.root)

    def get_yoloformat_list(self):
        return self.yoloformat_list

    def parse(self):
        w, h = self.get_size()
        yoloformat = YOLOFormat()
        # yoloformat.w = 1
        # yoloformat.h = 1
        path = self.get_path()
        yoloformat.set_image_path(path)

        object_list = self.root.findall(Keys.object)
        try:
            for obj in object_list:
                obj_info = ObjectInfo()
                obj_info, name = self.get_category_name(obj=obj, obj_info=obj_info)
                # self.categroy_counter.counter[name] += 1
                obj_info = self.get_coordinate(obj=obj, obj_info=obj_info, w=w, h=h)

                # print(obj_info.categroy_id)
                # print(obj_info.xmax)
                yoloformat.add_obj_info(obj=obj_info)
            self.yoloformat_list.append(yoloformat)
        except KeyError as e:
            print(e)
            pass

    def get_path(self):
        path = self.root.find('path')
        return path.text

    def get_category_name(self, obj, obj_info):
        name = obj.find(Keys.name)
        # id = self.categroy_map.mapping[name.text]
        id = self.categroy_dict[name.text]
        obj_info.categroy_id = id


        return obj_info, name.text

    def get_coordinate(self, obj, obj_info, w, h):
        bndbox = obj.find(Keys.bndbox)
        xmin = bndbox.find(Keys.xmin)
        ymin = bndbox.find(Keys.ymin)
        xmax = bndbox.find(Keys.xmax)
        ymax = bndbox.find(Keys.ymax)

        # xmin = int(xmin.text) / obj_info.w
        # ymin = int(ymin.text) / obj_info.h
        # xmax = int(xmax.text) / obj_info.w
        # ymax = int(ymax.text) / obj_info.h
        xmin = int(xmin.text)
        ymin = int(ymin.text)
        xmax = int(xmax.text)
        ymax = int(ymax.text)

        width = xmax - xmin
        height = ymax - ymin

        center_x = (xmax + xmin) / 2.0
        center_y = (ymax + ymin) / 2.0

        width = width / w
        height = height / h
        center_x = center_x / w
        center_y = center_y / h


        obj_info.x = str(center_x)
        obj_info.y = str(center_y)
        obj_info.w = str(width)
        obj_info.h = str(height)
        return obj_info

        # return xmin, ymin, xmax, ymax

    def get_size(self):
        size = self.root.find(Keys.size)
        w = size.find(Keys.width)
        h = size.find(Keys.height)
        w = int(w.text)
        h = int(h.text)
        return w, h

    def path_join(self, path_list):
        path = path_list[0]
        for i in range(len(path_list)-1):
            path = path + '/' + path_list[i+1]
        return path



    def write_annotation_txt(self, annotation_path, yolo_format):
        with open(annotation_path, 'w') as f:
            for obj_info in yolo_format.obj_info_list:
                f.writelines(obj_info.categroy_id + ' ')
                f.writelines(obj_info.x + ' ')
                f.writelines(obj_info.y + ' ')
                f.writelines(obj_info.w + ' ')
                f.writelines(obj_info.h + '\n')


    def write_images_path_txt(self, images_path, yolo_format):
        with open(images_path, 'a') as f:
            f.writelines(yolo_format.image_path + '\n')

    def make_yolo_text_format(self, output_folder):
        print('sss', self.yoloformat_list)
        # label_dir = 'labels'
        label_dir = output_folder
        images_path = 'train.txt'
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        for yolo_format in self.yoloformat_list:
            print(yolo_format.image_path)
            path = yolo_format.image_path.split('/')
            filename = path[-1]
            filename = filename[:-4]
            print(filename)


            # path = re.sub('\', '', yolo_format.image_path)
            # print(path)
            # p = re.compile(r'\w\S\S\w*\S')

            # res = self.pattern.search(yolo_format.image_path).group()
            annotation_path = '{}.{}'.format(self.path_join([label_dir, filename]), 'txt')
            print('annotation_path', annotation_path)

            self.write_annotation_txt(annotation_path=annotation_path, yolo_format=yolo_format)
            # self.write_images_path_txt(images_path=images_path, yolo_format=yolo_format)

    def show_categroy_counter(self):
        for (key, val) in self.categroy_counter.counter.items():
            print(key, val)

    def run(self):
        """
        run the parser
        :return: None
        """
        annotation_list = os.listdir(self.annotation_folder)
        for path in annotation_list:
            print(path)
            full_path = '{}/{}'.format(self.annotation_folder, path)
            print(full_path)
            self.get_root(xml_path=full_path)
            self.parse()

        self.make_yolo_text_format(output_folder=self.output_folder)




def main():
    # annotation_path = 'annotation/'
    # annotation_folder = '/media/taka/dataset/label_preprocess_data/frames/reiver_data/labels_xml'
    # output_folder = '/media/taka/dataset/label_preprocess_data/frames/reiver_data/test'
    annotation_folder = '/media/taka/dataset/pillar_and_bouy/test_video/pillar/_labels'
    output_folder = '/media/taka/dataset/pillar_and_bouy/test_video/pillar/labels'

    categroy_dict = {
        'pillar': '0',
        'buoy': '1'
    }
    # categroy_counter = CategroyCounter()
    xml_parser = Xml2YoloParser(
        categroy_dict=categroy_dict, annotation_folder=annotation_folder,
        output_folder=output_folder)

    xml_parser.run()

    # annotation_list = os.listdir(annotation_folder)
    # for path in annotation_list:
    #     print(path)
    #     full_path = '{}/{}'.format(annotation_folder, path)
    #     print(full_path)
    #     xml_parser.get_root(xml_path=full_path)
    #     xml_parser.parse()
    #
    # xml_parser.make_yolo_text_format(output_folder=output_folder)
    # xml_parser.show_categroy_counter()




if __name__ == "__main__":
    main()
