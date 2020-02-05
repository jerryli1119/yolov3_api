
import metric_module
import os
from progressbar import ProgressBar


class ObjectDetectionMetric:
    def __init__(self, predict_folder, label_list, coco_names):
        self.predict_folder = predict_folder
        self.label_list = label_list
        self.coco_name = coco_names
        self.metric = self._metrix_init()
        self.THRESH_CONFIDENCE = 0.1
        self.THRESH_IOU_CONFUSION = 0.5

    def _metrix_init(self):
        with open(self.coco_name) as f:
            NAMES_CLASS = f.read().splitlines()
        # NUMBER_CLASSES = len(NAMES_CLASS)

        ###
        # print("# of data: %d" % len(IMAGENAMES_GROUNDTRUTH))
        metric = metric_module.ObjectDetectionMetric(names_class=NAMES_CLASS,
                                                     check_class_first=False)
        return metric

    def run(self):
        DICT_TEXTNAMES_PREDICTION = \
            {os.path.splitext(p)[0]: os.path.join(self.predict_folder, p) for p in
                                     os.listdir(self.predict_folder)}
        # print(DICT_TEXTNAMES_PREDICTION)

        with open(self.label_list) as f:
            IMAGENAMES_GROUNDTRUTH = f.read().splitlines()
        # print(IMAGENAMES_GROUNDTRUTH)

        # THRESH_CONFIDENCE = 0.1
        # THRESH_IOU_CONFUSION = 0.5

        # NAMES_CLASS = [str(n) for n in range(80)]


        # label_generator = labels_module.LabelGenerator(number_color = NUMBER_CLASSES+1)
        # image = cv2.imread(IMAGENAMES_GROUNDTRUTH[0])
        # height_image, width_image = image.shape[:2]
        # label_generator.get_legend(size=3,
        #                            names_class=NAMES_CLASS,
        #                            height_image=height_image,
        #                            width_image=width_image)

        pbar = ProgressBar().start()
        for index in range(len(IMAGENAMES_GROUNDTRUTH)):
            imagename = IMAGENAMES_GROUNDTRUTH[index]
            textname_prediction = DICT_TEXTNAMES_PREDICTION[os.path.splitext(os.path.basename(imagename))[0]]
            textname_groundtruth = imagename.replace("images", "labels").replace("jpg", "txt")

            with open(textname_groundtruth) as f:
                info_groundtruth = f.read().splitlines()
            bboxes_groundtruth = []
            labels_groundtruth = []
            for bbox in info_groundtruth:
                bbox = bbox.split()
                label = int(bbox[0])
                # label = 0
                bboxes_groundtruth.append([float(c) for c in bbox[1:5]])
                labels_groundtruth.append(label)

            with open(textname_prediction) as f:
                info_prediction = f.read().splitlines()
            bboxes_prediction = []
            labels_prediction = []
            scores_prediction = []
            for bbox in info_prediction:
                bbox = bbox.split()
                label = int(bbox[0])
                # label      = 0
                confidence = float(bbox[5])
                if confidence >= self.THRESH_CONFIDENCE:
                    bboxes_prediction.append([float(c) for c in bbox[1:5]])
                    labels_prediction.append(label)
                    scores_prediction.append(confidence)

            self.metric.update(bboxes_prediction=bboxes_prediction,
                          labels_prediction=labels_prediction,
                          scores_prediction=scores_prediction,
                          bboxes_groundtruth=bboxes_groundtruth,
                          labels_groundtruth=labels_groundtruth)
            progress = 100 * index / len(IMAGENAMES_GROUNDTRUTH)
            pbar.update(progress)
        pbar.finish()

    def get_voc07(self):
        self.metric.get_mAP(type_mAP="VOC07",
                       conclude=True)

    def get_voc12(self):
        self.metric.get_mAP(type_mAP="VOC12",
                       conclude=True)

    def get_coco(self):
        self.metric.get_mAP(type_mAP="COCO",
                       conclude=True)

    def get_confusion(self):
        self.metric.get_confusion(thresh_confidence=self.THRESH_CONFIDENCE,
                             thresh_IOU=self.THRESH_IOU_CONFUSION,
                             conclude=True)
def main():
    predict_folder = '/home/taka/river_data/cropped/predicts'
    label_list = '/home/taka/river_data/cropped/data_list/test.txt'
    coco_names = '/home/taka/river_data/cropped/coco.names'
    odm = ObjectDetectionMetric(predict_folder=predict_folder, label_list=label_list,
                                coco_names=coco_names)
    odm.run()
    odm.get_voc07()
    odm.get_confusion()

if __name__ == '__main__':
    main()