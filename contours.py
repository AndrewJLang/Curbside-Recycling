import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("sample_images/tennis_balls/video_10_frames/image00004.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

plt.imshow(thresh)
plt.show()