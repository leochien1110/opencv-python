#!/usr/bin/env python3
import cv2
import numpy as np


#read source image
img=cv2.imread('res/rail06.png')

#convert sourece image to HSC color mode
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV tuning
hsv_low = np.array([92, 8, 122], np.uint8)
hsv_high = np.array([107, 13, 142], np.uint8)

#making mask for hsv range
mask = cv2.inRange(hsv, hsv_low, hsv_high)
cv2.imshow('HSV Image', mask)

# erode
kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations = 1)
erosion = cv2.erode(dilation, kernel, iterations = 1)
cv2.imshow('Dilated Image', dilation)
cv2.imshow('Eroded Image', erosion)

# edge detection
canny = cv2.Canny(mask, 150, 150)
cv2.imshow('Canny Image', canny)

# Hough Lines Detection
minLineLength = 250
maxLineGap = 40
lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, 420, minLineLength, maxLineGap)
print(len(lines))
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('Line Detection', img)

cv2.waitKey(0)
cv2.destroyAllWindows()