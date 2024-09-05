#  feekra baset
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# preparing the data

input_directory = '/data/clf-data'
categories = ['empty', 'not_empty']

data = []
label = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_directory, category)):
        image_path = os.path.join(input_directory, category, file)
        image = imread(image_path)
        image = resize(image, (15, 15))
        data.append(image.flatten())  # to make an array
        label.append(category_idx)

data = np.asarray(data)
label = np.asarray(label)

# training / test spliting
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, shuffle=True, stratify=label)

# training classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.001], 'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier,parameters)

grid_search.fit(x_train, y_train)

# testing performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of Samples Arere Correctly Classified'.format(str(score * 100)))

# pickle.dump(best_estimator, open('./model.p', 'wb'))

