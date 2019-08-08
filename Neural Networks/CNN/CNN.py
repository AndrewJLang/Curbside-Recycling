import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

batch_size = 32
epochs = 1000

#currently going to form so that I am using handwritten data set provided by keras (adjust later to work for picture data set)
(trainingData, trainingLabels), (validationData, validationLabels) = mnist.load_data()

trainingLabels = to_categorical(trainingLabels)
validationLabels = to_categorical(validationLabels)

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

