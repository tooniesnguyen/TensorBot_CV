from mmdeploy_runtime import Detector
import cv2

img = cv2.imread('/home/toonies/Learn/TensorBot-Vision/mmdetection/demo/demo.jpg')

# create a detector
detector = Detector(model_path='/home/toonies/Learn/TensorBot-Vision/checkpoints/epoch_20.pth', device_name='cpu')
# run the inference
bboxes, labels, _ = detector(img)
# Filter the result according to threshold
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int),  bbox[4]
  if score < 0.3:
      continue
  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)