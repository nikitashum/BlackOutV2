import cv2
from time import sleep

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, photo = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", photo)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    sleep(1)
    cv2.imshow("test", photo)
    img_counter += 1

cam.release()

cv2.destroyAllWindows()


def get_frame():
    camera = cv2.VideoCapture(0)
    check, frame = camera.read()
    if not check:
        print("failed to grab frame")
        cam.release()
        return
    return frame
