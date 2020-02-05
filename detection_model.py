import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import numpy as np
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')

parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder

parser.add_argument('--img-size', type=int, default=1600, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
opt = parser.parse_args()

class detection_model(object):

    def __init__(self):
        self.model, self.device = self.detect()
        self.img_size = opt.img_size
        self.names = load_classes(opt.names)
    
    def _load_weight(self, weights, device, model):
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)


    def detect(self, save_img=False):
        # Initialize
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

        # Initialize model
        model = Darknet(opt.cfg, 100)

        # Load weights
        self._load_weight(weights=opt.weights, device=device, model=model)

        # Fuse Conv2d + BatchNorm2d layers
        model.fuse()

        # Eval mode
        model.to(device).eval()

        return model, device

    def _img_preprocess(self, img):

        img, _, _ = letterbox(img, new_shape=self.img_size)
        #self.yolo_shape_img = img.copy()
        #print('self.yolo_shape_img', self.yolo_shape_img.shape)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img

    def _parse_result(self, det, img, input_img):
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            print(img.shape, input_img.shape)
            scaled_xyxy = scale_coords(img.shape[2:], det[:, :4], input_img.shape).round()
            
            det[:, :4] = scaled_xyxy

            res_list = []
            for *xyxy, conf, cls_conf, cls in det:

                b_box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2])]
                print("\n", b_box,"\n")
                res = {
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "category": self.names[int(cls)],
                    "confidence": float(conf),
                }

                res_list.append(res)

            return res_list

    def inference_mode(self, input_img):
        img = copy.copy(input_img)
        img = self._img_preprocess(input_img)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)

        det = non_max_suppression(pred)[0]
        res_list = self._parse_result(det=det, img=img, input_img=input_img)

        return res_list


if __name__ == '__main__':
    yolov3 = detection_model()

    with torch.no_grad():
        yolov3.detect()
