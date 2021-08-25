#!/usr/bin/env python3

import time
import cv2

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
video.set(cv2.CAP_PROP_FPS, 30)

start = time.time()
while (True):

    ret, frame = video.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end = time.time()
    elapse = end - start
    print(1/elapse)
    start = end

video.release()

cv2.destroyAllWindows()
