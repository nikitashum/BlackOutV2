import os
import cv2
import numpy as np

IMG_WIDTH = 50
IMG_HEIGHT = 50


def create_dataset():

    data = "./DataSet"
    img_data_array = []
    class_name = []

    for directory in os.listdir(data):
        for file in os.listdir(os.path.join(data, directory)):
            image_path = os.path.join(data, directory,  file)
            image = cv2.imread(image_path, 0)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(file[16])
    return img_data_array, class_name
