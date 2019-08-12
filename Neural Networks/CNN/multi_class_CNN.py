import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
from keras_preprocessing import image
import matplotlib.pyplot as plt
import os
import csv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=True)
ap.add_argument("-b", "--batch", required=True)
args = vars(ap.parse_args())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = "../../pymeanshift/pms_images/"

#shuffle the arrays so they aren't in the same 'linear' order that they are in the csv file
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
    return data

#Extracts the labels for each of the images
def multiClassImgs():
    labels = []
    for x in os.listdir(data_dir):
        if "plastic" in x:
            labels.append(0)
            continue
        elif "soda" in x:
            labels.append(1)
            continue
        else:
            labels.append(2)
            continue
    return labels

data = np.array(processImgs())
labels = to_categorical(multiClassImgs())
data, labels = shuffle_arrays(data,labels)
split = int(len(data) * 0.8)

trainingData, trainingLabels = data[:split], labels[:split]
validationData, validationLabels = data[split:], labels[split:]

num_classes = len(labels[0])
batchSize = int(args['batch'])
epoch = int(args['epochs'])
learningrate = 0.001

imgShape = trainingData[0].shape

#CNN to run the images through
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=imgShape))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(lr=learningrate), loss=categorical_crossentropy, metrics=['accuracy'])

model.fit(trainingData, trainingLabels, batch_size=batchSize, epochs=epoch, verbose=1)

predictionArr = model.predict_classes(validationData).reshape(-1)

_, accuracy = model.evaluate(validationData, validationLabels, verbose=1)

# print(validationLabels.shape)

validationLabels = np.argmax(validationLabels, axis=1)

count = 0
for x in range(len(validationLabels)):
   print(f"Correct value: {validationLabels[x]}")
   if validationLabels[x] == predictionArr[x]:
       count += 1

#Both accuracies should be the same
print(f"Test validation accuracy: {float(count/len(validationLabels))}")

print(f"Model validation accuracy: {accuracy}")
