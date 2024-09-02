import os
import cv2

output_dir = './clf-data/all '
mask_path = './mask.png'
mask = cv2.imread(mask_path, 0)

analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

(totalLabels, Label_ids, values, centroid) = analysis

slots = []
for i in range(1, totalLabels):
    # Area of the component
    area = values[i, cv2.CC_STATE_AREA]
    # Coordinate of the bounding points
    x1 = values[i, cv2.CC_STAT_LEFT]
    y1 = values[i, cv2.CC_STAT_TOP]
    w = values[i, cv2.CC_STAT_WIDTH]
    h = values[i, cv2.CC_STAT_HEIGHT]
    # Coordinate of the bounding box
    pt1 = (x1, y1)
    pt2 = (x1 + w, y1 + h)
    (X, Y) = centroid[i]

    slots.append([x1, y1, w, h])

video_path = './data/parkinglot.mp4'
capture = cv2.VideoCapture(video_path)


frame_number = 00
capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = capture.read()

if ret:
    # frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))

    for slot_number, slot in enumerate(slots):
        if slot_number in [132, 147, 164, 180, 344, 360, 377, 341, 360, 179, 131, 186, 91, 61, 0,
                           4, 89, 129, 161, 185, 281, 224, 271, 383, 319, 335, 351, 389, 29, 12, 32, 72, 281, 280, 157,
                           223, 26]:
            slot = frame[slot[1]:slot[1] + slot[3], slot[0]:slot[0] + slot[2], :]

            cv2.imwrite(os.path.join(output_dir, '{}_{}.jpg'.format(str(frame_number).zfill(0), str(slot_number).zfill(8))), slot)
            frame_number += 10

            capture.release()