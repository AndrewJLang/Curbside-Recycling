import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

#Attempt to use Canny edge detection for object recognition, not effective atm

img = cv2.imread("sample_images/soda_cans/video_5_frames/image00004.jpg")
edges = cv2.Canny(img, 100,200)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(edges, cmap='gray')

plt.show()