import time
import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
from RRTStar_New import find_rrt_path
import datetime
import multiprocessing
import matplotlib.pyplot as plt

######################################################################
width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
deadZone = 55  # initially it was 100
startCounter = 0
global imgContour
isHeightCorrect = 0
######################################################################



########################

frameWidth = width
frameHeight = height


def create_white_image(width=640, height=480, filename="final_Path.jpg"):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    plt.imsave(filename, image)


def detect_objects(img):
    # Get img shape
    model = YOLO('yolo-weights/yolov8x-seg.pt')
    height, width, channels = img.shape
    results = model.predict(source=img.copy(), save=False, save_txt=False, retina_masks=True)
    result = results[0]
    masks = result.masks.cpu().data.numpy()
    summed_masks = np.sum(masks, axis=0)
    normalized_mask = 255 - (summed_masks > 0).astype(np.uint8) * 255
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
    return bboxes, class_ids, segmentation_contours_idx, scores, normalized_mask


def calculate_rrt_path(img, start_point):
    end_point = None
    if start_point[0] is None:
        print(" user is not detected")
        return 0

    if img is None:
        print("Input image is None. Exiting function.")
        return 0

    try:
        bboxes, classes, segmentations, scores, masks_img = detect_objects(img)
        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            try:
                # our destination goal is a bicycle, if it's found within the frame, save its coordinate. This is our destination point.
                # truck=7 | bicycle=1 | traffic light=9
                if class_id == 1:
                    end_point = tuple(np.min(seg, axis=0))
                cv2.fillPoly(img, [seg], (0, 0, 0))
            except Exception as e_inner:
                print(f"Error while processing object: {e_inner}")
        find_rrt_path(masks_img, start_point, end_point)

    except Exception as e_outer:
        print(f"Error in draw_contour_on_objects function: {e_outer}")


def empty(a):
    pass


def stack_images(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def control_drone(img, imgContour, me):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    global isHeightCorrect
    cx=None
    cy=None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area < areaMin:
            continue
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # print(len(approx))
        x, y, w, h = cv2.boundingRect(approx)
        cx = int(x + (w / 2))  # CENTER X OF THE OBJECT
        cy = int(y + (h / 2))  # CENTER Y OF THE OBJECT
        if isHeightCorrect:
            # there might be another objects with the same user's shirt color, if their center point isn't near the dead-zone, it's likely not the user.
            if cx > (frameWidth/2 + 2*deadZone) or cx < (frameWidth/2 - 2*deadZone):
                continue
            if cy < (frameHeight - 4 *deadZone):
                continue
            if cy < (frameHeight - 2*deadZone-20):
                cv2.putText(imgContour, "FORWARD", (int(frameWidth/2 - deadZone+20), int(frameHeight/2 - 2*deadZone)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
                me.for_back_velocity = 35

            elif cy > (frameHeight - deadZone-10):
                cv2.putText(imgContour, "BACKWARD", (int(frameWidth/2 - deadZone+10), int(frameHeight/2 + 2*deadZone)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
                me.for_back_velocity = -35

            else:
                me.for_back_velocity = 0


            if cx < int(frameWidth / 2) - deadZone:

                cv2.putText(imgContour, " <- ROTATE LEFT <-", (10, int(frameHeight/2)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
                me.yaw_velocity = -30

            elif cx > int(frameWidth / 2) + deadZone:

                cv2.putText(imgContour, "-> ROTATE RIGHT ->", (int(frameWidth/2 + deadZone)+10, int(frameHeight/2)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
                me.yaw_velocity = 30

            else:
                me.yaw_velocity = 0
            cv2.circle(imgContour, (cx, cy), 4, (0, 255, 0), 3)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            break

    height = me.get_height()
    if height < 400 and not isHeightCorrect:
        me.up_down_velocity = 40
        cv2.putText(imgContour, "GO UP || height is: " + str(height), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    (255, 0, 0), 2)

    elif height > 410 and not isHeightCorrect:
        me.up_down_velocity = -30
        cv2.putText(imgContour, "GO DOWN || height is: " + str(height), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    (255, 0, 0), 2)

    else:
        me.up_down_velocity = 0
        isHeightCorrect = 1

    cy2=min(cy+90,479) if cy is not None else None
    cx2=cx+20 if cx is not None else None
    return me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity,cy2,cx2


def display(img):
    cv2.line(img, (int(frameWidth / 2) - deadZone, int(frameHeight) - 3*deadZone), (int(frameWidth / 2) - deadZone, frameHeight-deadZone), (0, 255, 255), 3)
    cv2.line(img, (int(frameWidth / 2) + deadZone, int(frameHeight) - 3*deadZone), (int(frameWidth / 2) + deadZone, int(frameHeight) - deadZone), (0, 255, 255), 3)
    cv2.line(img, (int(frameWidth / 2) - deadZone, int(frameHeight) - deadZone), (int(frameWidth / 2) + deadZone, int(frameHeight) - deadZone), (0, 255, 255), 3)
    cv2.line(img, (int(frameWidth / 2) - deadZone, int(frameHeight) - 3*deadZone), (int(frameWidth / 2) + deadZone, int(frameHeight) - 3*deadZone), (0, 255, 255), 3)

def create_cv_windows():
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 640, 240)
    cv2.createTrackbar("HUE Min", "HSV", 117, 179, empty)
    cv2.createTrackbar("HUE Max", "HSV", 132, 179, empty)
    cv2.createTrackbar("SAT Min", "HSV", 208, 255, empty)
    cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
    cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("VALUE Max", "HSV", 210, 255, empty)

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 640, 240)
    cv2.createTrackbar("Threshold1", "Parameters", 89, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 0, 255, empty)
    cv2.createTrackbar("Area", "Parameters", 1400, 7000, empty)

    cv2.namedWindow("RRT* Path Planning")
    cv2.resizeWindow("RRT* Path Planning", 640, 480)


def get_trackbar_values():
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    return h_min, h_max, s_min, s_max, v_min, v_max


def preprocess_image(frame, width, height):
    img = cv2.resize(frame, (width, height))
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, imgHsv


def filter_image(img, lower, upper):
    mask = cv2.inRange(img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return result, mask


def process_image(img):
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    return imgGray


def detect_edges(img, threshold1, threshold2):
    imgCanny = cv2.Canny(img, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    return imgDil

def get_thresholds():
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    return threshold1, threshold2


def worker(args):
    # Your worker function logic here
    objectsImage, start_point = args
    calculate_rrt_path(objectsImage, start_point)


def main():
    global startCounter
    pool = multiprocessing.Pool()
    # CONNECT TO TELLO
    drone = Tello()
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0
    drone.TIME_BTW_RC_CONTROL_COMMANDS = 0.1

    print(drone.get_battery())

    drone.streamoff()
    drone.streamon()
    create_cv_windows()
    create_white_image()
    frame = None
    while frame is None or frame.mean() < 10:  # Check for black frame
        frame = drone.get_frame_read().frame
        print("NO FRAME!")
        time.sleep(0.1)

    execution_time = time.time()
    while True:
        # GET THE IMAGE FROM TELLO
        frame_read = drone.get_frame_read()
        myFrame = frame_read.frame

        img, imgHsv = preprocess_image(myFrame, width, height)
        imgContour = img.copy()
        objectsImage = img.copy()

        h_min, h_max, s_min, s_max, v_min, v_max = get_trackbar_values()
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        result, mask = filter_image(imgHsv, lower, upper)
        imgGray = process_image(result)

        threshold1, threshold2 = get_thresholds()
        imgDil = detect_edges(imgGray, threshold1, threshold2)

        drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity,cx,cy = control_drone(imgDil, imgContour, drone)
        display(imgContour)

        current_time = time.time()
        if current_time - execution_time >= 10 and isHeightCorrect:
            execution_time = current_time
            start_point = cx,cy
            print(start_point)
            pool.apply_async(worker, args=((objectsImage, start_point),))

        ################# FLIGHT

        if startCounter == 0:
            drone.takeoff()
            time.sleep(1)
            startCounter = 1

        # SEND VELOCITY VALUES TO TELLO

        if drone.send_rc_control:
            drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity)

        # stack = stack_images(0.9, ([img, result], [imgDil, imgContour]))
        # cv2.imshow('Horizontal Stacking', stack)
        image_bgr = cv2.cvtColor(imgContour, cv2.COLOR_RGB2BGR)
        cv2.imshow('user track', image_bgr)
        try:
            rrt_image = cv2.imread("final_path.jpg")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if rrt_image is None:
                continue
            cv2.putText(rrt_image, f'{timestamp} | Battery: {drone.get_battery()}', (10, rrt_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)
            cv2.imshow("RRT* Path Planning", rrt_image)

        except FileNotFoundError as e:
            print(e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.land()
            drone.end()

            break

    cv2.destroyAllWindows()
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()