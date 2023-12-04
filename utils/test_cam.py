import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)


pipe.start(cfg)

while True:
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    ir_frame = frame.get_infrared_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())

    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                     alpha = 0.5), cv2.COLORMAP_JET)

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    x, y = 640, 360
    cv2.circle(color_image, (x, y), 10, (0,255,0),2)

    cv2.imshow('rgb', color_image)
    cv2.imshow('depth', depth_cm)
    cv2.imshow('ir', ir_image)

    if cv2.waitKey(1) == ord('q'):
        break
pipe.stop()

