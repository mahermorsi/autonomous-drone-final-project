import numpy as np
import cv2
from ultralytics import YOLO
from RRTStar_New import find_rrt_path
from mask_white import convert_non_black_to_white


def get_start_point_from_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_x = float('inf')
    min_y = float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        area_min = cv2.getTrackbarPos("Area", "Parameters")
        if area < area_min:
            continue
        for point in contour:
            x, y = point[0]
            min_x = min(min_x, x)
            min_y = min(min_y, y)

    # If no points found, return None
    if min_x == float('inf') or min_y == float('inf'):
        return None

    # Return the smallest x and y coordinates found
    return min_x, min_y


def detect_objects(img):
    # Get img shape
    model = YOLO('../yolo-weights/yolov8s-seg.pt')
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

    # bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    # Get class ids
    # class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
    # Get scores
    # scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
    # return bboxes, class_ids, segmentation_contours_idx, scores
    return segmentation_contours_idx

def calculate_rrt_path(img, start_point):
    if img is None:
        print("Input image is None. Exiting function.")
        return 0

    try:
        # bboxes, classes, segmentations, scores = detect_objects(img)
        # for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        segmentations = detect_objects(img)
        for seg in segmentations:
            try:
                # (x, y, x2, y2) = bbox
                # cv2.polylines(img, [seg], True, (0, 0, 255), 4)
                cv2.fillPoly(img, [seg], (0, 0, 0))
                # cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            except Exception as e_inner:
                print(f"Error while processing object: {e_inner}")

        drone_frame_path = "../image results/captured_image.jpg"
        cv2.imwrite(drone_frame_path, img)
        filtered_image_path = convert_non_black_to_white(drone_frame_path)
        if filtered_image_path:
            find_rrt_path(filtered_image_path, start_point)
        else:
            print("convert_non_black_to_white returned None. Unable to proceed.")

    except Exception as e_outer:
        print(f"Error in draw_contour_on_objects function: {e_outer}")


img = cv2.imread("../image results/drone frame.jpg")
imgDil = cv2.imread("../image results/contour.jpg")
start_point = get_start_point_from_contour(imgDil)
calculate_rrt_path(img,start_point)

