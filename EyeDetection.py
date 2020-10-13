import cv2
import numpy as np
from CaptureFeed import get_frame

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
# save the image(i) in the same directory
img = get_frame()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

dim = (50, 50)

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    resized = cv2.resize(roi_color[ey:ey+eh, ex:ex+ew], dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('img'+str(ex), resized)
    print(np.shape(resized))
cv2.waitKey(0)
cv2.destroyAllWindows()
