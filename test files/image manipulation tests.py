import cv2
import numpy as np

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


def get_start_point_from_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_x = float('inf')
    min_y = float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        area_min = 1000
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

width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
myFrame=cv2.imread('../example screenshots/pic3.png')

img, imgHsv = preprocess_image(myFrame, width, height)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.imshow("image", imgHsv)
cv2.waitKey(0)

h_min, h_max, s_min, s_max, v_min, v_max = 117, 132, 208, 255, 0, 210
lower = np.array([h_min, s_min, v_min])
upper = np.array([h_max, s_max, v_max])

result, mask = filter_image(imgHsv, lower, upper)
cv2.imshow("image", result)
cv2.waitKey(0)
cv2.imshow("image", mask)
cv2.waitKey(0)
imgGray = process_image(result)
cv2.imshow("image", imgGray)
cv2.waitKey(0)

threshold1 = 89
threshold2 = 0
imgDil = detect_edges(imgGray, threshold1, threshold2)
cv2.imshow("image", imgDil)
cv2.waitKey(0)
start = get_start_point_from_contour(imgDil)
print(start)
cv2.circle(imgDil, tuple(start), 5, (255, 0, 0), -1)
cv2.imshow("image", imgDil)
cv2.waitKey(0)


