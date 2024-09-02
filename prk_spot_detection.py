# feekra baset
import cv2
import numpy as np
from util import get_parking_spot_boxes, zero_or_not


def calc_diff(im1, im2):  # Compute the value how different the two spots are
    return np.abs(np.mean(im1) - np.mean(im2))  # for absolute number shortform abs


mask_path = './mask_crop.png'

video_path = './data/parking_crop.mp4'

mask = cv2.imread(mask_path, 0)

capture = cv2.VideoCapture(video_path)

connected_compnts = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spot_boxes(connected_compnts)

spots_status = [None for j in spots]

diffs = [None for j in spots]

previous_frame = None

frame_nmr = 0
ret = True
step = 30  # how much time needed for next classification
while ret:
    ret, frame = capture.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            a3, b3, w, h = spot

            spot_crp = frame[b3:b3 + h, a3:a3 + w, :]

            diffs[spot_indx] = calc_diff(spot_crp, previous_frame[b3:b3 + h, a3:a3 + w, :])

        print(diffs)
    if frame_nmr % step == 0:
        for spot_indx, spot in enumerate(spots):  # calling the rectangle from using binary mask
            a3, b3, w, h = spot

            spot_crp = frame[b3:b3 + h, a3:a3 + w, :]

            spot_status = zero_or_not(spot_crp)

            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]

            a3, b3, w, h = spots[spot_indx]

    if spot_status:
        frame = cv2.rectangle(frame, (a3, b3), (a3 + w, b3 + h), (0, 255, 0), 2)
    else:
        frame = cv2.rectangle(frame, (a3, b3), (a3 + w, b3 + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('y'):
        break

    frame_nmr += 1

capture.release()
cv2.destroyAllWindows()
