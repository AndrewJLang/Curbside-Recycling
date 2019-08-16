#Constants for the CNN

import numpy as np
from keras_preprocessing import image

data_dir = "../../pymeanshift/pms_images/"

#Extracts the images and places them in an array
def processImgs():
    data = []
    for x in os.listdir(data_dir):
        img = image.load_img(data_dir + x, target_size=(400,400))
        img = image.img_to_array(img)
        img = img / 255
        data.append(img)
    return data

def shuffle_arrays(arr1, arr2):
    assert len(arr1) == len(arr2)
    shuffle_arr1 = np.empty(arr1.shape, dtype=arr1.dtype)
    shuffle_arr2 = np.empty(arr2.shape, dtype=arr2.dtype)
    perm = np.random.permutation(len(arr1))
    for oldIndex, newIndex in enumerate(perm):
        shuffle_arr1[newIndex] = arr1[oldIndex]
        shuffle_arr2[newIndex] = arr2[oldIndex]
    return shuffle_arr1, shuffle_arr2

def labelIndividualData(objectType):
    labelArr = []
    objectType = str(objectType)
    for x in os.listdir(data_dir):
        if objectType in x:
            labelArr.append(1)
        else:
            labelArr.append(0)
    return np.array(labelArr)

def getTensors(arr):
    tensorArr = []
    for x in range(len(arr)):
        featureVector = np.reshape(arr[x], newshape=-1)
        tensorArr.append(featureVector)
    return tensorArr


#Variables
objectDict = {
    'ball': 0,
    'can': 1,
    'bottle': 2,
    'paper': 3,
    'other': 4
}

FILTERCOUNT = 16
STRIDES = (1,1)
BALLKERNEL = (2,2)
BOTTLEKERNEL = (2,2)
CANKERNEL = (2,2)
PAPERKERNEL = (2,2)
OTHERKERNEL = (2,2)