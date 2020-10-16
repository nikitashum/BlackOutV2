from EyeDetection import get_eyes
import cv2
from time import sleep

for x in range(6):
    eyes = get_eyes()
    print(eyes)
    cv2.imshow("left", eyes[0])
    cv2.imshow("right", eyes[1])
    sleep(1)

cv2.VideoCapture.release()
