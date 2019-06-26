import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

#Attempt to use Canny edge detection for object recognition, not effective atm

capture = cv2.VideoCapture("blue_background_sample_images/slow_motion/tennis_balls/IMG_4214_Slomo.mp4")

_, img = capture.read()
edges = cv2.Canny(img, 100,200)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(edges, cmap='gray')

plt.show()

#Code from hsv file
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# img = cv2.imread("sample_images/plastic_bottles/video_6_frames/image00006.jpg")
img = cv2.imread("sample_images/plastic_bottles/video_2_frames/image00004.jpg")

rescaled = rescale_frame(img, 50)

hsv = cv2.cvtColor(rescaled, cv2.COLOR_RGB2HSV)
cv2.imshow('picture', hsv)
#Include following line if you wish to save the hsv image
# cv2.imwrite("image00004_hsv.jpg", hsv)

cv2.waitKey(0)