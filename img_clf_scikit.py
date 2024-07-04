# feekra
import os

from skimage.io import imread
from skimage.transform import resize
import numpy as np
# prepare data

input_directory = 'C:/Users/Baishakhi/PycharmProjects/myPython/opencv_python/imageClassification/data'
categories = ['empty', 'not_empty']

data = []
label = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_directory, category)):
        image_path = os.path.join(input_directory, category, file)
        image = imread(image_path)
        image = resize(image, (15, 16))
        data.append(image.flatten())
        label.append(category_idx)

data = np.asarray(data)
label = np.asarray(label)

# train/ test split
# test performance
