import cv2
from track import detect_objects
import matplotlib.pyplot as plt
import numpy as np
from RRTStar_New import find_rrt_path

img = cv2.imread("../example screenshots/pic3.png")
start_point = 300, 300
bboxes, classes, segmentations, scores, mask_img = detect_objects(img)
for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):

    # our destination goal is a bicycle, if it's found within the frame, take a point. This is our destination point.
    if class_id == 1:
        end_point = tuple(np.min(seg, axis=0))
        print(end_point)
    cv2.fillPoly(img, [seg], (0, 0, 0))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# resized_image = cv2.resize(mask_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
# cv2.imshow('winname',resized_image)
# plt.show()
find_rrt_path(mask_img,start_point,end_point)

