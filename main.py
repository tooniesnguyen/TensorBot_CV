import pyrealsense2 as rs
import numpy as np
import cv2
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
import mmdet
from mmdet.apis import DetInferencer
import sys
import socket
import os
from pathlib import Path


HOST = socket.gethostbyname(socket.gethostname())
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WORK_DIR = os.path.dirname(ROOT)

class TRACKING:
    def __init__(self, config_path, checkpoint):
        self.inferencer = DetInferencer(model=config_path, weights=checkpoint, device='cpu')
        

    def detect_person(self, img):
        result = self.inferencer(img)['predictions'][0]['bboxes']
        return result
    
    @staticmethod
    def realsense_init():
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
        pipe.start(cfg)

        return pipe, cfg

    
    def realsense_start(self, pipe, cfg):
        bb_ready = 0
        mode_CSRT = 0
        tracker = cv2.legacy.TrackerCSRT_create()

        while True:
            frame = pipe.wait_for_frames()
            color_frame = frame.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            if (not bb_ready) and (not mode_CSRT):
                result = self.detect_person(color_image)
                if result:
                    print("############################# Run Mode Detect #############################")
                    # [[355.6026916503906, 255.78961181640625, 460.9337158203125, 452.4129943847656]]
                    x1,y1,x2,y2 = int(result[0][0]), int(result[0][1]), int(result[0][2]), int(result[0][3])

                    #Convert x1,y1,x2,y2 to x,y,h,w

                    x,y,h,w = x1,y1, y2-y1, x2-x1
                    cv2.rectangle(color_image, (x1,y1), (x2, y2), (0,0,255), 2 )
                    bb_ready = 1
            elif bb_ready:
                print("############################# Run Mode Start #############################")
                ret = tracker.init(color_image, (x,y,w,h))
                mode_CSRT = 1
                bb_ready = 0
                
            elif mode_CSRT:
                ret, obj = tracker.update(color_image)
                print("############################# Run Mode CSRT #############################")
                if ret:
                    p1 = (int(obj[0]), int(obj[1]))
                    p2 = (int(obj[0]+obj[2]), int(obj[1]+obj[3]))
                    cv2.rectangle(color_image, p1, p2, (0, 255, 0), 2)
                else:
                    mode_CSRT = 0

            cv2.imshow('rgb', color_image)
            if cv2.waitKey(1) == ord('q'):
                break
        pipe.stop()
        


def main():
    config_path = os.path.join(WORK_DIR, "TensorBot-Vision/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py")
    checkpoint = os.path.join(WORK_DIR, "TensorBot-Vision/checkpoints/epoch_1.pth")
    track = TRACKING(config_path, checkpoint)
    pipe, config = track.realsense_init()


    track.realsense_start(pipe, config)
    


if __name__ == "__main__":
    main()
