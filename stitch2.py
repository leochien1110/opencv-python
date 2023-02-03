import stitching
import cv2

settings = {"detector": "sift", "confidence_threshold": 0.2}

stitcher = stitching.Stitcher(**settings)
panorama = stitcher.stitch(["res/1.jpg", "res/2.jpg", "res/3.jpg"])

cv2.imshow('panorama', panorama)
cv2.waitKey(0)