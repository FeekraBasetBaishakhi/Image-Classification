import pickle

from skimage.transform import resize
import numpy as np
import cv2

zero = True
one = False

model = pickle.load(open("model.p", "rb"))


def zero_or_not(spot_bgr):
    flat_data = []

    image_resize = resize(spot_bgr, (15, 15, 3))
    flat_data.append(image_resize.flatten())
    flat_data = np.array(flat_data)

    y_output = model.predict(flat_data)

    if y_output == 0:
        return zero
    else:
        return one


def get_parking_spot_boxes(connected_compnts):
    (totaLabels, label_ids, values, centric) = connected_compnts

    slot = []
    coof = 1

    for i in range(1, totaLabels):
        # coordinate points
        a3 = int(values[i, cv2.CC_STAT_LEFT] * coof)
        b3 = int(values[i, cv2.CC_STAT_TOP] * coof)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coof)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coof)

        slot.append([a3, b3, w, h])

    return slot
