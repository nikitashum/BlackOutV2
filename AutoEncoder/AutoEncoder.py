from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import random

data_folder = "../DataSet"

plt.figure(figsize=(20, 20))

for i in range(5):
    directory = random.choice(os.listdir(data_folder))
    dir_path = os.path.join(data_folder, directory)
    image = random.choice(os.listdir(dir_path))
    image_path = os.path.join(dir_path, image)
    img = mpimg.imread(image_path)
    ax = plt.subplot(1, 5, i+1)
    image_text = "Open" if image[16] == "1" else "Close"
    ax.title.set_text(image_text)
    plt.imshow(img)
plt.show()
