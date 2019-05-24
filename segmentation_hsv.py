import cv2
import numpy as np

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# img = cv2.imread("sample_images/plastic_bottles/video_6_frames/image00006.jpg")
img = cv2.imread("sample_images/soda_cans/video_10_frames/image00003.jpg")

rescaled = rescale_frame(img, 50)

hsv = cv2.cvtColor(rescaled, cv2.COLOR_RGB2HSV)
cv2.imshow('picture', hsv)

cv2.waitKey(0)
