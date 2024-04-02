import cv2


def display(img):
    cv2.line(img, (int(frameWidth / 2) - deadZone, int(frameHeight) - 3*deadZone), (int(frameWidth / 2) - deadZone, frameHeight-deadZone), (0, 255, 255), 3)
    cv2.line(img, (int(frameWidth / 2) + deadZone, int(frameHeight) - 3*deadZone), (int(frameWidth / 2) + deadZone, int(frameHeight) - deadZone), (0, 255, 255), 3)
    cv2.line(img, (int(frameWidth / 2) - deadZone, int(frameHeight) - deadZone), (int(frameWidth / 2) + deadZone, int(frameHeight) - deadZone), (0, 255, 255), 3)
    cv2.line(img, (int(frameWidth / 2) - deadZone, int(frameHeight) - 3*deadZone), (int(frameWidth / 2) + deadZone, int(frameHeight) - 3*deadZone), (0, 255, 255), 3)


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