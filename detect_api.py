import argparse

from queue import Queue
from flask import Flask, app, jsonify, request, make_response
from flask_restful import Api, Resource
from gevent.pywsgi import WSGIServer
from threading import Thread, Lock
from yolo_utils_v2.box_drawer import BoxDrawer, EmptyBoxDrawer
import cv2
import sys

# Synchronize to Asynchronize
from gevent import monkey
monkey.patch_all(thread=False)

import requests # dependency with monkey

import time
import datetime

# yolo model
from detection_model import detection_model
'''
# load config 
from cfg.model_cfg import detect_cfg
# yolo cfg dict change to class format
from utils.detect_utils import dict_to_class
'''
# byte to image
from yolo_utils_v2.data_loader.format_utils import ByteToImg

#cfg = dict_to_class(detect_cfg)
model = detection_model()

class YoloMode(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
            
        # run yolo ...............................................................................
        img = ByteToImg.__call__(data)
        res_list = model.inference_mode(input_img=img)

        if not res_list:
            res_list = None

        print(res_list)
        print('yolo_done')
        sys.stdout.flush()
        return res_list, img


app = Flask(__name__)
api = Api(app)

yolo_mode = YoloMode()

class ImageData:
        # For display
        frame_q = Queue()

class Detect(Resource):
    def __init__(self):
        pass

    def post(self):
        try:
            data = request.data

            print('yolo mode')
            res_list, img = yolo_mode(data=data)
            print('    res_list ', res_list)
            sys.stdout.flush()

            frame_data = {'img': data, 'res': res_list}
            # For display
            if False:
                ImageData.frame_q.put(frame_data)

            if res_list is None:
                res_list = []
            else:
                for r in res_list:
                    r['x1'] = r['box'][0]
                    r['y1'] = r['box'][1]
                    r['x2'] = r['box'][2]
                    r['y2'] = r['box'][3]

            res = jsonify({'res': res_list})

        except Exception as e:

            print('error', e)
            sys.stdout.flush()

            raise Exception
            app.logger.error(e)

            res_list = None
            frame_data = {'img': data, 'res': res_list}

            # For display
            if False:
                ImageData.frame_q.put(frame_data)

            res = jsonify({'res': 'fail'})

        return res


class DisplayWorker(Thread):
    def __init__(self):
        #super().__init__()
        Thread.__init__(self)
        self.frame_drawer = BoxDrawer()

    def run(self):
        cv2.namedWindow('show1')
        while True:
            frame_data = ImageData.frame_q.get()
            
            img = frame_data['img']
            res = frame_data['res']
            img = ByteToImg.__call__(img)
            print('img', img.shape, 'res', res)
            sys.stdout.flush()
            
            if res is not None:
                img = self.frame_drawer.draw_box_with_file(box_list=res, img=img)
            
            t1 = time.time()
            cv2.imshow('show1', img)
            print("draw   ",time.time()-t1)
            sys.stdout.flush()
            k = cv2.waitKey(10)

            if k == ord(' '):
                cv2.destroyWindow('show1')
                break

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5566)
    opt = parser.parse_args()

    api.add_resource(Detect, '/detect')

    http_server = WSGIServer(('0.0.0.0', opt.port), app)
    http_server.serve_forever()

if __name__ == "__main__":

    #display_worker = DisplayWorker()
    #display_worker.start()
    main()

