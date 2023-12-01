import pyrealsense2 as rs
import numpy as np
import cv2
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
import mmdet
from mmdet.apis import DetInferencer


config_path = '/home/robotino/Desktop/Nhan_CDT/TensorBot-Vision/mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
checkpoint = '/home/robotino/Desktop/Nhan_CDT/TensorBot-Vision/checkpoints/epoch_1.pth'
inferencer = DetInferencer(model=config_path, weights=checkpoint, device='cpu')
pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

pipe.start(cfg)

while True:
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                     alpha = 0.5), cv2.COLORMAP_JET)

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    result = inferencer(color_image)
    print(result)
    cv2.imshow('rgb', color_image)
    cv2.imshow('depth', depth_cm)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()