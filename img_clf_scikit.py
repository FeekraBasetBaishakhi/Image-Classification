# feekra baset
import os

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split

# prepare data

input_directory = 'C:/Users/Baishakhi/PycharmProjects/myPython/imageClassification/data'
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
a_train, a_test, y_train, y_test = train_test_split(data, label, test_size = 0.3, shuffle= True, stratify= label)
# test performance
