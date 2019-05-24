import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# image = cv2.imread('sample_images/plastic_bottles/video_6_frames/image00006.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# image = image.reshape((image.shape[0] * image.shape[1], 3))
# cluster = KMeans(n_clusters=6)
# cluster.fit(image)


# plt.figure()
# plt.imshow(image)
# plt.show()

image = cv2.imread('sample_images/soda_cans/video_10_frames/image00004.jpg')/255
# plt.figure(figsize=(15,8))
# plt.imshow(image)

x, y, z = image.shape
image_shape = image.reshape(x*y, z)
# print(x)

k_means = KMeans(n_clusters=8)
k_means.fit(image_shape)
cluster_centers = k_means.cluster_centers_
cluster_labels = k_means.labels_

plt.figure(figsize=(15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))

plt.show()