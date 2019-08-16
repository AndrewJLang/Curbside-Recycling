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

#python modules
import constants

"""
This class is for creating individual CNN models for each data set
From here, their tensors will be combined in order to form one large data set to
put into/train on a NN
"""

data = constants.processImgs()
labels = constants.createLabels()
imgShape = data[0].shape

ballData, bottleData, canData, paperData, backgroundData = constants.separateDataTypes(data, labels)

#Ball model
modelBall = Sequential()
modelBall.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.BALLKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
ballCompiled = modelBall.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
ballPredictions = modelBall.predict(ballData)
print(ballPredictions.shape)

#Can model
modelCan = Sequential()
modelCan.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.CANKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
canCompiled = modelCan.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
canPredictions = modelCan.predict(canData)
print(canPredictions.shape)

#Bottle model
modelBottle = Sequential()
modelBottle.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.BOTTLEKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
bottleCompiled = modelBottle.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
bottlePredictions = modelBottle.predict(bottleData)
print(bottlePredictions.shape)

#Paper model
modelPaper = Sequential()
modelPaper.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.PAPERKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
paperCompiled = modelPaper.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
paperPredictions = modelPaper.predict(paperData)
print(paperPredictions.shape)

#Background/other model
modelBackground = Sequential()
modelBackground.add(Conv2D(filters=constants.FILTERCOUNT, kernel_size=constants.BACKGROUNDKERNEL, strides=constants.STRIDES, padding=constants.PADDING, activation='relu', bias_initializer=RandomNormal(), input_shape=imgShape))
backgroundCompiled = modelBackground.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
backgroundPredictions = modelBackground.predict(backgroundData)
print(backgroundPredictions.shape)

"""
Now we will combine the predictions in order to learn off these values
We need to turn these 'predictions' into 1d arrays of tensor values
These will be an array containing the amount of images ran through (x) and the feature vector = length x width x filter (y)
"""

ballTensor = constants.getTensors(ballPredictions)
print(ballTensor.shape)

canTensor = constants.getTensors(canPredictions)
print(canTensor.shape)

bottleTensor = constants.getTensors(bottlePredictions)
print(bottleTensor.shape)

paperTensor = constants.getTensors(paperPredictions)
print(paperTensor.shape)

backgroundTensor = constants.getTensors(backgroundPredictions)
print(backgroundTensor.shape)

#Now we must concatenate all these separate tensors so they can be run through a NN
tensor = np.concatenate((ballTensor, canTensor, bottleTensor, paperTensor, backgroundTensor), axis=0)
print(tensor.shape)

#Need to shuffle data before feeding in to network
tensor, labels = constants.shuffle_arrays(tensor, labels)
print(f"After shuffle shape-tensor: {tensor.shape}\tlabels: {labels.shape}")

#create the model for collective data, 
finalModel = Sequential()
finalModel.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])
finalModel.add(Dense(64, activation='relu', bias_initializer=RandomNormal(), input_shape=tensor.shape))
finalModel.add(Dropout(0.2))

#final layer of NN
finalModel.add(Dense(constants.class_count, activation='softmax'))
finalModel.compile(optimizer=Adam(lr=constants.learningrate), loss=categorical_crossentropy, metrics=['accuracy'])

#Get a summary of the model so far
finalModel.summary()

#Now to fit the data to the model and train/validate it
fittedModel = finalModel.fit(tensor, labels, batch_size=constants.finalLayerBatchSize, epochs=constants.EPOCHCOUNT, verbose=1, validation_split=0.2)

#Now to display the graph/chart of both the training and validation accuracy + loss