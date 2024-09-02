import pickle

from skimage.transform import resize
import numpy as np
import cv2

zero = True
one = False

MODEL = pickle.load(open("model.p", "rb"))


def zero_or_not(spot_bgr):
    """

    @rtype: object
    """
    flat_data = []

    image_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(image_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return zero
    else:
        return one


def get_parking_spot_boxes(connected_compnts):
    (totalLabels, label_ids, values, centric) = connected_compnts

    slots = []
    coef = 1

    for i in range(1, totalLabels):
        # coordinate points
        a3 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        b3 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([a3, b3, w, h])

    return slots
