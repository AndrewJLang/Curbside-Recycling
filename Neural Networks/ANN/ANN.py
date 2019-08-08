import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import argparse
import csv

#use to specify the activation function one desires to use for the hidden layers
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--activation", required=True)
ap.add_argument("-l", "--layers", required=True)
ap.add_argument("-o", "--outputlayer", required=True)
ap.add_argument("-w", "--writeOutput", required=True)
args = vars(ap.parse_args())

dataset =  np.loadtxt("./CSV Files/Dataset.csv", delimiter=",")
nanIndex = np.argwhere(np.isnan(dataset))

#removes the 'nan' items from the array
dataset = dataset[~np.any(np.isnan(dataset), axis=1)]

truthValues = np.loadtxt("./CSV Files/GroundTruths.csv", delimiter=",")

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

#removes the 'nan' item positions from the labels
nanIndex = np.delete(nanIndex, 1, 1)
nanIndex = np.unique(nanIndex)
truthValues = np.delete(truthValues, nanIndex)
labels = to_categorical(truthValues)

dataset, labels = shuffle_arrays(dataset, labels)

for x in range(len(dataset)):
    for i in range(len(dataset[x])):
        dataset[x][i] = float(dataset[x][i] / 255)

trainingLength = int(len(dataset) * 0.8)
trainingData = dataset[:trainingLength]
validationData = dataset[trainingLength:]
trainingLabels = labels[:trainingLength]
validationLabels = labels[trainingLength:]

tf.logging.set_verbosity(tf.logging.ERROR)
activationFunc = args["activation"]
layerCount = args["layers"]
outputLayer = int(args["outputlayer"])


#create the layers for the NN
model = Sequential()
model.add(Dense(100,input_shape=(3,), activation=activationFunc))

for x in range(int(layerCount)):
    model.add(Dense(outputLayer, activation=activationFunc))

model.add(Dense(4, activation='softmax'))

# Stochastic = optimizers.SGD(lr=0.1)

#learning rate at 0.1 does not perform well (gets stuck in local minimum)
learningRate = 0.01

model.compile(optimizer=optimizers.Adam(lr=learningRate), loss=losses.categorical_crossentropy, metrics=['accuracy'])

modelFit = model.fit(dataset, labels, epochs=1000, batch_size=744, verbose=2, validation_split=0.2)

# score = model.evaluate(validationData, validationLabels, batch_size=15)

model.summary()

# print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

def writeCSV(model,outputLayers, layerCount, epochs, batchSize, opt="Adam", activation="relu", learningRate=0.01):
        TrainingAcc = model.history['acc']
        trainingLoss =  model.history['loss']
        ValidationAcc = model.history['val_acc']
        validationLoss = model.history['val_loss']
        with open("CSV Files/ANNResults.csv", mode='a', newline='') as f:
                fileWriter = csv.writer(f)
                f.write("Epochs: " + str(epochs) + "     BatchSize: " + str(batchSize) + "     Learning Rate: " + str(learningRate))
                f.write("\nOptimizer: " + opt + "     Activation: " + activation)
                f.write("\nTraining Acc: " + str(TrainingAcc[-1]) + "     Validation Acc: " + str(ValidationAcc[-1]))
                f.write("\nTraining Loss: " + str(trainingLoss[-1]) + "     Validation Loss: " + str(validationLoss[-1]))
                f.write("\n\n")

writeOutput = args["writeOutput"]
if int(writeOutput) == 1:
        writeCSV(modelFit, outputLayer, layerCount, epochs=1000, batchSize=744, learningRate=0.01)

#graph's both the training and validation data to see performance
def model_data(model):
    fig, axs = plt.subplots(2,2, figsize=(12,8), constrained_layout=True)
    axs[0][0].plot(range(1, len(model.history['acc'])+1), model.history['acc'])
    axs[0][0].set_title('Training Accuracy')
    axs[0][0].set_ylabel('Accuracy')
    axs[0][0].set_xlabel('Epoch')
    axs[0][0].set_xticks(np.arange(1,len(model.history['acc'])+1),len(model.history['acc'])/10)

    axs[0][1].plot(range(1, len(model.history['val_acc'])+1), model.history['val_acc'])
    axs[0][1].set_title('Validation Accuracy')
    axs[0][1].set_ylabel('Accuracy')
    axs[0][1].set_xlabel('Epoch')
    axs[0][1].set_xticks(np.arange(1,len(model.history['val_acc'])+1),len(model.history['val_acc'])/10)

    axs[1][0].plot(range(1, len(model.history['loss'])+1), model.history['loss'])
    axs[1][0].set_ylabel('Loss')
    axs[1][0].set_title('Training Loss')
    axs[1][0].set_xlabel('Epoch')
    axs[1][0].set_xticks(np.arange(1, len(model.history['loss'])+1), len(model.history['loss'])/10)
    
    axs[1][1].plot(range(1, len(model.history['val_loss'])+1), model.history['val_loss'])
    axs[1][1].set_ylabel('Loss')
    axs[1][1].set_title('Validation Loss')
    axs[1][1].set_xlabel('Epoch')
    axs[1][1].set_xticks(np.arange(1, len(model.history['val_loss'])+1), len(model.history['val_loss'])/10)

    plt.show()



model_data(modelFit)