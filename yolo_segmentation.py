from ultralytics import YOLO
import cv2
import numpy as np


# cap = cv2.VideoCapture(0)
def detect(img):
    # Get img shape
    model = YOLO("../yolo-weights/yolov8m-seg.pt")
    height, width, channels = img.shape
    results = model.predict(source=img.copy(), save=False, save_txt=False)
    result = results[0]
    segmentation_contours_idx = []
    for seg in result.masks.xyn:
        # contours
        seg[:, 0] *= width
        seg[:, 1] *= height
        segment = np.array(seg, dtype=np.int32)
        segmentation_contours_idx.append(segment)

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    # Get class ids
    class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
    # Get scores
    scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
    return bboxes, class_ids, segmentation_contours_idx, scores


def draw_contour_on_objects(img):
    bboxes, classes, segmentations, scores = detect(img)
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        # (x, y, x2, y2) = bbox
        # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        cv2.polylines(img, [seg], True, (0, 0, 255), 4)
        # cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow("Objects contour", img)
    cv2.waitKey(1)


# while True:
#
#     success, img = cap.read()
#     draw_contour_on_objects(img)
