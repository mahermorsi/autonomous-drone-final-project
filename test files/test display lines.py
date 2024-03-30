import cv2


def display(img):
    # cv2.line(img, (int(frameWidth / 2) - deadZone, 0), (int(frameWidth / 2) - deadZone, frameHeight), (0, 255, 255), 3)
    # cv2.line(img, (int(frameWidth / 2) - 2 * deadZone, 0), (int(frameWidth / 2) - 2 * deadZone, frameHeight),
    #          (0, 0, 255), 3)
    # cv2.line(img, (int(frameWidth / 2) + deadZone, 0), (int(frameWidth / 2) + deadZone, frameHeight), (0, 255, 255), 3)
    # cv2.line(img, (int(frameWidth / 2) + 2 * deadZone, 0), (int(frameWidth / 2) + 2 * deadZone, frameHeight),
    #          (0, 0, 255), 3)
    cv2.line(img, (0, int(frameHeight) - deadZone), (frameWidth, int(frameHeight) - deadZone), (0, 255, 255), 3)
    cv2.line(img, (0, int(frameHeight) - 3*deadZone), (frameWidth, int(frameHeight) - 3*deadZone), (0, 255, 255), 3)
    cv2.line(img, (0, frameHeight -4*deadZone), (frameWidth, frameHeight-4*deadZone), (255, 0, 0), 3)



deadZone = 40
width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
frameWidth = width
frameHeight = height
img = cv2.imread("../example screenshots/captured_image.jpg")
cv2.imshow("test2",img)
img = cv2.resize(img, (width, height))


display(img)
cv2.imshow("test",img)
cv2.waitKey(0)