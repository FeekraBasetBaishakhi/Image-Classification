# feekra baset
import cv2

from util import get_parking_spot_boxes


mask_path = './mask.png'

video_path = './data/park.mp4'


mask = cv2.imread(mask_path, 0)

capture = cv2.VideoCapture(video_path)

connected_compnts = cv2.connectedComponentsWithStats(mask, 3, cv2.CV_32S)

spots = get_parking_spot_boxes(connected_compnts)
ret = True

while ret:
    ret, frame = capture.read()
    for spot in spots:
        a3, b3, w, h = spot

        frame = cv2.rectangle(frame,(a3,b3), (a3 + w, b3 + h), (0, 0, 255), 4)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('y'):
        break

capture.release()
cv2.destroyAllWindows()






