import cv2


def get_frame():
    camera = cv2.VideoCapture(0)
    check, frame = camera.read()
    if not check:
        print("failed to grab frame")
        camera.release()
        return
    return frame
