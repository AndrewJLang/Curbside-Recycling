import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
from keras_preprocessing import image
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import os
import csv
import argparse
import sys

np.set_printoptions(threshold=sys.maxsize)

# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--epochs", required=True)
# ap.add_argument("-b", "--batch", required=True)
# args = vars(ap.parse_args())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = "../../pymeanshift/pms_mixed_frames/"

objectDict = {
    'ball': 0,
    'can': 1,
    'bottle': 2,
    'paper': 3,
    'other': 4
}

#Going to create labels for the images by placing them in an array
def createLabels():
    labelArr = []
    objectTypes = ['ball', 'can', 'bottle', 'paper']
    for x in os.listdir(data_dir):
        positions = np.zeros(5)
        for i in range(len(objectTypes)):
            if objectTypes[i] in x:
                positions[i] = 1
        if np.count_nonzero(positions) == 0:
            positions[4] = 1
        labelArr.append(positions)
    return np.array(labelArr)

def shuffle_arrays(arr1, arr2):
    assert len(arr1) == len(arr2)
    shuffle_arr1 = np.empty(arr1.shape, dtype=arr1.dtype)
    shuffle_arr2 = np.empty(arr2.shape, dtype=arr2.dtype)
    perm = np.random.permutation(len(arr1))
    for oldIndex, newIndex in enumerate(perm):
        shuffle_arr1[newIndex] = arr1[oldIndex]
        shuffle_arr2[newIndex] = arr2[oldIndex]
    return shuffle_arr1, shuffle_arr2

#Extracts the images and places them in an array
def processImgs():
    data = []
    for x in os.listdir(data_dir):
        img = image.load_img(data_dir + x, target_size=(400,400))
        img = image.img_to_array(img)
        img = img / 255
        data.append(img)
    return np.array(data)

data = processImgs()
labels = createLabels()

def separateDataTypes(data, labels):
    ballData, bottleData, canData, paperData, backgroundData = [], [], [], [], []
    for x in range(len(data)):
        if labels[x][0] == 1:
            ballData.append(data[x])
        if labels[x][1] == 1:
            bottleData.append(data[x])
        if labels[x][2] == 1:
            canData.append(data[x])
        if labels[x][3] == 1:
            paperData.append(data[x])
        if labels[x][4] == 1:
            backgroundData.append(data[x])
    return np.array(ballData), np.array(bottleData), np.array(canData), np.array(paperData), np.array(backgroundData)

ballData, bottleData, canData, paperData, backgroundData = separateDataTypes(data, labels)

#Define the image shape that will be fed through CNN, constant for all data sets as defined within processImgs method
imgShape = ballData[0].shape


#create the models for each batch of images

#QUESTION: Do I need to create a separate model for each object?
modelBall = Sequential()
modelBall.add(Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
ballConv = modelBall.predict(ballData)
print(ballConv.shape)

modelBottle = Sequential()
modelBottle.add(Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
bottleConv = modelBottle.predict(bottleData)
print(f"bottle shape after convolution: {bottleConv.shape}")

#This is going to extract the tensors from each filter, making it one large 1d array
#Say if the array is size 13x399x399x16 (13 images, 399 width, 399 height, 16 filters used), then it turns each image from 3d to 1d. Ex. 399*399*16 is new tensor array length, with it being 1d
def extractVectors(arr):
    tensorArr = []
    for x in range(len(arr)):
        featureVector = np.reshape(arr[x], newshape=-1)
        tensorArr.append(featureVector)
    return np.array(tensorArr)

#Now to extract the tensor for each object/image
ballTensor = extractVectors(ballConv)
bottleTensor = extractVectors(bottleConv)
print(ballTensor.shape)
print(bottleTensor.shape)

#Properly concatenates lists
#tensor = np.concatenate((ballTensor,bottleTensor), axis=0)

labels = [objectDict['ball'] for x in range(len(ballTensor))]
labels.extend([objectDict['bottle'] for x in range(len(bottleTensor))])
print(labels)

newlab = to_categorical(labels)
print(newlab)



