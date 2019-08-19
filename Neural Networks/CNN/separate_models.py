import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.activations import sigmoid
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
from keras_preprocessing import image
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import os
import csv
import argparse
import sys

#python modules
import constants

"""
This class is for creating individual CNN models for each data set
From here, their tensors will be combined in order to form one large data set to
put into/train on a NN
"""

#The entire data set
data = constants.processImgs()

#Define the shape of each image (will all be the same)
imgShape = data[0].shape

"""
Since each CNN will be trained separately of one another, each will be a binary classification
Because of this, the CNN's will have binary labels encoded to each of their individual models 
"""

ballLabels = constants.labelIndividualData('ball')
canLabels = constants.labelIndividualData('can')
bottleLabels = constants.labelIndividualData('bottle')
paperLabels = constants.labelIndividualData('paper')
backgroundLabels = constants.labelIndividualData('other')

"""
Now we must separate the data for training/validation for each CNN and the final NN
Note that the validation data will not be run in conjunction to the training CNN
The separation will be done using an 80/20 split
"""
splitMark = int(len(data) * 0.8)

trainingData, validationData = data[:splitMark], data[splitMark:]
ballTrainingLabels, ballValidationLabels = data[:splitMark], data[splitMark:]
canTrainingLabels, canValidationLabels = data[:splitMark], data[splitMark:]
bottleTrainingLabels, bottleValidationLabels = data[:splitMark], data[splitMark:]
paperTrainingLabels, paperValidationLabels = data[:splitMark], data[splitMark:]
backgroundTrainingLabels, backgroundValidationLabels = data[:splitMark], data[splitMark:]

print(np.array(trainingData).shape)

#NOTE: I need to learn how to save the models for easier test validation

#Ball model
modelBall = Sequential()
modelBall.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.BALLKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=(199,199,3)))
modelBall.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
modelBall.fit(np.array(trainingData), np.array(ballTrainingLabels), batch_size=constants.BALLBATCHSIZE, epochs=constants.EPOCHCOUNT, verbose=1)


# #Can model
# modelCan = Sequential()
# modelCan.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.CANKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
# canCompiled = modelCan.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
# canPredictions = modelCan.predict(canData)
# print(canPredictions.shape)

# #Bottle model
# modelBottle = Sequential()
# modelBottle.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.BOTTLEKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
# bottleCompiled = modelBottle.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
# bottlePredictions = modelBottle.predict(bottleData)
# print(bottlePredictions.shape)

# #Paper model
# modelPaper = Sequential()
# modelPaper.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.PAPERKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
# paperCompiled = modelPaper.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
# paperPredictions = modelPaper.predict(paperData)
# print(paperPredictions.shape)

# #Background/other model
# modelBackground = Sequential()
# modelBackground.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.BACKGROUNDKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
# backgroundCompiled = modelBackground.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
# backgroundPredictions = modelBackground.predict(backgroundData)
# print(backgroundPredictions.shape)

"""
Now we will combine the predictions in order to learn off these values
We need to turn these 'predictions' into 1d arrays of tensor values
These will be an array containing the amount of images ran through (x) and the feature vector = length x width x filter (y)
"""
