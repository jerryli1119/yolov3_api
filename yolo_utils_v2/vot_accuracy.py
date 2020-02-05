from yolo_utils_v2.iou import get_IOU
from yolo_utils_v2.data_loader.image_loader import ImageLoaderFromFolder
from yolo_utils_v2.parser_helper import LabelParser


class VOTAccuracy:
    """
    get the average of iou of the gt and predict file
    if the predict is none, the bbox is (0,0,0,0)
    example:
    images_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/images'
    labels_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/labels'
    predicts_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/predicts'
    vot_acc = VOTAccuracy(images_folder=images_folder,
                          labels_folder=labels_folder, predicts_folder=predicts_folder)
    vot_acc.run()
    """
    def __init__(self, images_folder, labels_folder, predicts_folder):
        """
        :param images_folder:
        :param labels_folder:
        :param predicts_folder:
        """
        self.labels_folder = labels_folder
        self.predicts_folder = predicts_folder
        self.data_loader = ImageLoaderFromFolder(folder=images_folder)
        self.label_parser = LabelParser()

    def run(self):
        """
        get the average iou
        :return: average iou
        """
        total_iou = 0
        frame_iter = self.data_loader.frame_iter()
        iter_length = 0

        for i, (img, filename) in enumerate(frame_iter):
            h, w, _ = img.shape
            label_path = '{}/{}.txt'.format(self.labels_folder, filename)
            predict_path = '{}/{}.txt'.format(self.predicts_folder, filename)
            label_list = self.label_parser.parse_label(
                label_path=label_path, img_w=w, img_h=h
            )
            predict_list = self.label_parser.parse_label(
                label_path=predict_path, img_w=w, img_h=h
            )
            for gt, predict in zip(label_list, predict_list):
                # print('gtgt', gt, 'pppp', predict)
                iou = get_IOU(bbox1=gt['box'], bbox2=predict['box'])
                # print(iou)
                total_iou += iou
            iter_length = i+1

        acc = total_iou / iter_length
        return acc


def main():
    images_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/images'
    labels_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/labels'
    predicts_folder = '/media/taka/dataset/pillar_and_bouy/test_video/buoy_far/predicts'
    vot_acc = VOTAccuracy(images_folder=images_folder,
                          labels_folder=labels_folder, predicts_folder=predicts_folder)
    vot_acc.run()


if __name__ == "__main__":
    main()

