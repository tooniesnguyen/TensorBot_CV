import cv2
import numpy as np

# Reduce fps
import time

cap = cv2.VideoCapture("../images/test_tracking.mp4")


if not cap.isOpened():
    print("Can not open camera")
    exit()


# Step 1: Create tracker
# tracker type: Boosting, MedianFlow, MIL, TLD, MOSSE, KCF, CSRT
tracker = cv2.legacy.TrackerCSRT_create()

# Step 2: Detect objec
ret, frame = cap.read()
r = cv2.selectROI(frame) # return (x, y, w, h)
# initialize tracker
ret = tracker.init(frame, r)



while True:


    prev = time.time()

    # read frame by frame
    ret, frame = cap.read()
    if not ret:
        print(" Cant read video")
        break

    # Step 3: update tracker
    ret, obj = tracker.update(frame)
    # obj: (x, y, w, h)
    if ret:
        # draw bounding box
        p1 = (int(obj[0]), int(obj[1]))
        p2 = (int(obj[0]+obj[2]), int(obj[1]+obj[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    else:
        print("tracking fail")

    

    cv2.imshow("Tracking", frame)

    # close
    if cv2.waitKey() == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()