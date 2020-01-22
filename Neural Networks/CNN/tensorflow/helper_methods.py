import numpy as np
import random
import tensorflow as tf
import sys
import os
import cv2

#import python module
import constants

np.set_printoptions(threshold=sys.maxsize)


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

    # print(labelArr[:batchSize])
    #Need to transform labels to have each row contain binary values for which object image contains
    #NOTE: I need to not have the images separated when doing this if they have multiple labels
    labelArr = tf.one_hot(labelArr, 5).eval()
    # print(f"This is label array: {np.array(labelArr.eval())}")

    return np.array(imgArr[:batchSize]), np.array(labelArr[:batchSize])

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

"""
This will be the function called when optimizing the data
Need the directories of all the images of the individual objects, as well as mixed group
Split data up into batches and formatted so that testing can be done
"""
def getBatchInfo(batchSize, category='mixed'):
    directory = [constants.ball_dir, constants.bottle_dir, constants.can_dir, constants.paper_dir, constants.background_dir]
    categories = [constants.ball_ONEHOT, constants.bottle_ONEHOT, constants.can_ONEHOT, constants.paper_ONEHOT, constants.background_ONEHOT]

    batchFiles, imageData, imageLabels = [], [], []

    #Check this code
    for imageDir in directory:
        if os.path.exists(imageDir):
            tmp = os.listdir(imageDir)
            a = random.randint(0, len(tmp)-1)
            path = os.path.join(imageDir, tmp[a])
            batchFiles.append(path)
        else:
            print("imagePath does not exist")

    if category == 'mixed':
        for i, f in enumerate(batchFiles):
            original_img = cv2.imread(f, cv2.IMREAD_COLOR)
            img = cv2.resize(original_img, (constants.FULLSIZE, constants.FULLSIZE),interpolation=cv2.INTER_CUBIC) #check if interpolation is necessary
            width, length, depth = img.shape

            for x in range(batchSize):
                low = int(constants.IMGSIZE / 2)
                high = int(width - (constants.IMGSIZE / 2)-1)
                a = random.randint(low, high)
                b = random.randint(low, high)
                lowBox1 = int(a-(constants.IMGSIZE /2))
                lowBox2 = int(b-(constants.IMGSIZE / 2))
                highBox1 = int(a + (constants.IMGSIZE /2))
                highBox2 = int(b + constants.IMGSIZE / 2))
                box = img[lowBox1:highBox1, lowBox2:highBox2]
                imageData.append(box)
                imageLabels.append(categories[i])
    
    elif category == 'ball' or category == 'bottle' or category == 'can' or category == 'paper' or category == 'background':
        index = directory.index(category)
        for i, f in enumerate(batchFiles):
            original_img = cv2.imread(f,cv2.IMREAD_COLOR)
            width, length, depth = img.shape

            if i == index:
                k = 1
            else:
                k = constants.CLASSCOUNT

            for j in range(int(n/k)):
                low = int(constants.IMGSIZE / 2)
                high = int(w - (constants.IMGSIZE / 2) - 1)
                a = random.randint(low,high)
                b = random.randint(low,high)
                lowBox1 = int(a - (constants.IMGSIZE / 2))
                lowBox2 = int(b - (constants.IMGSIZE / 2))
                highBox1 = int(a + (constants.IMGSIZE / 2))
                highBox2 = int(b + (constants.IMGSIZE / 2))

                box = img[lowBox1:highBox1,lowBox2:highBox2]
                imageData.append(box)
                if i == index:
                    imageLabels.append([1])
                else:
                    imageLabels.append([0])
        
    zipUp = list(zip(imageData, imageLabels))
    random.shuffle(zipUp)
    imageData, imageLabels = zip(*zipUp)

    return np.array(imageData)[:batchSize], np.array(imageLabels)[:batchSize]
