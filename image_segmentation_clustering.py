import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# image = plt.imread("sample_images/tennis_balls/video_10_frames/image00004.jpg")/255
# print(image.shape)
# # plt.imshow(image)

# picture = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
# print(picture.shape)

# kmeans = KMeans(n_clusters=6, random_state=0).fit(picture)
# newPicture = kmeans.cluster_centers_[kmeans.labels_]
# clusterPic = newPicture.reshape(image.shape[0],image.shape[1], image.shape[2])

# plt.imshow(clusterPic)



# plt.show()