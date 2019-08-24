import numpy as np
from keras_preprocessing import image
import os

data_dir = "../../../pymeanshift/pms_mixed_frames/"

#Extracts the images and places them in an array
def processImgs():
    data = []
    for x in os.listdir(data_dir):
        img = image.load_img(data_dir + x, target_size=(200,200))
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

#Variables
objectDict = {
    'ball': 0,
    'can': 1,
    'bottle': 2,
    'paper': 3,
    'other': 4
}

#This is done for the collective data set
BATCHSIZE = 20

BALLBATCHSIZE = 1
BACKGROUNDBATCHSIZE = 100



EPOCHCOUNT = 10
VERB=1
FILTERCOUNT = 3 #Don't know if this can be changed, because the the dimensionality of the images (width x height x depth, i.e. filter count?)
STRIDES = (1,1)
BALLKERNEL = (2,2)
BOTTLEKERNEL = (2,2)
CANKERNEL = (2,2)
PAPERKERNEL = (2,2)
BACKGROUNDKERNEL = (5,5)
learningrate = 0.001
PADDING='SAME'
class_count = 5 #ball, bottle, can, paper, background...can add cardboard later