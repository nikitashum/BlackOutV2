import cv2
import numpy as np
from CaptureFeed import get_frame


def get_eyes():
    eyes = []
    roi_gray = []
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    img = get_frame()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ret = []
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        image = cv2.resize(roi_gray[ey:ey+eh, ex:ex+ew], (50, 50), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        ret.append(image)
    return ret
