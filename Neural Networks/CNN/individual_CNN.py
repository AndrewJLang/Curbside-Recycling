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

def processImgs():
    data = []
    for x in os.listdir(data_dir):
        img = image.load_img(data_dir + x, target_size=(400,400))
        img = image.img_to_array(img)
        img = img / 255
        data.append(img)
    return np.array(data)

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

#Create the individual binary labels for whether a certain label is in the image
ballLabels = labelIndividualData('ball')
canLabels = labelIndividualData('can')
bottleLabels = labelIndividualData('bottle')
paperLabels = labelIndividualData('paper')
otherLabels = labelIndividualData('other')

#The labels for the data set as a whole
labels = createLabels()
#The data of the images
data = processImgs()

imgShape = data[0].shape

#shuffle data sets and labels
ballData, ballLabels = shuffle_arrays(data, ballLabels)


#individual CNN's for each image, fed into their own NN's
modelBall = Sequential()
modelBall.add(Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
ballOptimizer = modelBall.compile(optimizer=Adam(lr=0.01), loss=categorical_crossentropy, metrics=['accuracy'])
#Direct predicition with no training of this model
prediction = modelBall.predict_classes(ballData, batch_size=1)
print(prediction)

# modelBottle = Sequential()
# modelBottle.add(Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
# ballOptimizer = modelBottle.compile(optimizer=Adam(lr=0.01), loss=categorical_crossentropy, metrics=['accuracy'])
# #Direct predicition with no training of this model
# prediction = modelBottle.predict_classes(ballData, batch_size=1)
# print(prediction)