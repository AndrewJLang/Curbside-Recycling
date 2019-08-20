import numpy as np
import random

#import python module
import constants

def separateTraining(validationSplit=0.8):
    data = constants.processImgs()
    preprocessedLabels = constants.createLabels()
    collective = list(zip(data, preprocessedLabels))
    random.shuffle(collective)
    shuffledData, shuffledLabels = zip(*collective)
    splitMark = int(len(shuffledData) * validationSplit)
    trainingData, validationData = shuffledData[:splitMark], shuffledData[splitMark:]
    trainingLabels, validationLabels = shuffledLabels[:splitMark], shuffledLabels[splitMark:]
    return trainingData, trainingLabels, validationData, validationLabels

#Separates the data according to the given batch size so that training can be done
#NOTE: I think this needs to be changed to hold a to_categorical-like array
def getBatchData(batchSize, trainingData, trainingLabels):
    imgArr, labelArr = [], []
    assert len(trainingData) == len(trainingLabels) #Check to ensure they are the same length
    # print(trainingLabels.shape)
    for x in range(len(trainingData)):
        if np.count_nonzero(trainingLabels[x]) > 1:
            for i in range(len(trainingLabels[x])):
                if trainingLabels[x][i] == 1:
                    imgArr.append(trainingData[x])
                    labelArr.append(i)
            continue
        elif np.count_nonzero(trainingLabels[x]) == 1:
            imgArr.append(trainingData[x])
            for i in range(len(trainingLabels[x])):
                if trainingLabels[x][i] == 1:
                    labelArr.append(i)

    print(labelArr[:batchSize])
    return imgArr[:batchSize], labelArr[:batchSize]

#For the validation data, this will not be run in batches, so there is no need to specify batch size
def getValidationData(validationData, validationLabels):
    imgArr, labelArr = [], []
    assert len(trainingData) == len(trainingLabels) #Check to ensure they are the same length
    # print(trainingLabels.shape)
    for x in range(len(trainingData)):
        if np.count_nonzero(trainingLabels[x]) > 1:
            for i in range(len(trainingLabels[x])):
                if trainingLabels[x][i] == 1:
                    imgArr.append(trainingData[x])
                    labelArr.append(i)
            continue
        elif np.count_nonzero(trainingLabels[x]) == 1:
            imgArr.append(trainingData[x])
            for i in range(len(trainingLabels[x])):
                if trainingLabels[x][i] == 1:
                    labelArr.append(i)
                    continue

    return imgArr, labelArr