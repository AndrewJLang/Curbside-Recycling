from PIL import Image
import os
import numpy as np
import sys
import csv
import datetime

np.set_printoptions(threshold=sys.maxsize)

#im = Image.open(os.path.join(os.path.dirname(__file__), "mask_adjusted_images/cardboard/image00018.jpg"))
#pix = im.load()

#Working properly
def getBitArray(category, imageName):
    imagePath = "mask_adjusted_images/" + category + "/" + imageName
    bitImage = Image.open(os.path.join(os.path.dirname(__file__), imagePath))
    pixels = bitImage.load()
    arr = np.empty([bitImage.size[1], bitImage.size[0]], dtype=int)

    #Assert that the array is the same dimensions as image
    assert arr.shape[0] == bitImage.size[1]
    assert arr.shape[1] == bitImage.size[0]
    pixelSums = [0, 765]
    countObject = 0

    for x in range(bitImage.size[1]):
        for y in range(bitImage.size[0]):
            rgbVal = int(sum(list(pixels[y,x])))
            bitVal = pixelSums[min(range(len(pixelSums)), key = lambda i: abs(pixelSums[i] - rgbVal))]
            if bitVal == 0:
                countObject += 1
                arr[x][y] = 1
            else:
                arr[x][y] = 0

    return arr, countObject, arr.shape[0]*arr.shape[1]


#Working properly
def loadCSV(folderName, pmsParameters, categoryName, fileName):
    pmsArr = []
    uniqueList = []
    dataFile = open(folderName + "/" + pmsParameters + "/" + categoryName + "/" + fileName, 'r')
    reader = csv.reader(dataFile, delimiter=",")
    for row in reader:
        intRow = [int(i) for i in row]
        for x in intRow:
            if x not in uniqueList:
                uniqueList.append(x)
        pmsArr.append(intRow)

    return np.array(pmsArr)

#loadCSV("pms_csv", "sr10rr10md100", "cardboard","image00018.csv")

def bitWiseCompare(baseImage, pmsImage):
    assert len(baseImage) == len(pmsImage)
    assert len(baseImage[0]) == len(pmsImage[0])

    similarityDict = {}
    misClassifiedDict = {}

    for x in range(len(baseImage)):
        for i in range(len(baseImage[0])):
            if baseImage[x][i] == 0:
                if pmsImage[x][i] in misClassifiedDict:
                    misClassifiedDict[pmsImage[x][i]] += 1
                else:
                    misClassifiedDict[pmsImage[x][i]] = 1
            else:
                if pmsImage[x][i] in similarityDict:
                    similarityDict[pmsImage[x][i]] += 1
                else:
                    similarityDict[pmsImage[x][i]] = 1
    
    return similarityDict, misClassifiedDict

def calcDictAccuracy(dictObject):
    totalAcc, objectCount = 0, 0
    for key in dictObject:
        objectCount += 1
        totalAcc += dictObject[key]
    
    return float(totalAcc / objectCount)

def writeOutput(fileName, PMSParam, classificationScore):
    fileObj = open(fileName, 'a')
    fileObj.write(str(PMSParam) + ": \t\t" + classificationScore + "\n")

"""
This is run on every labeling within image in order to find the best one
We do not care what the labeling is, just that the true positive is maximized while false positive is minimized
This is normalized
"""
def calcCombinedAccuracy(classifiedDict, misClassifiedDict, blobPixelCount, nonBlobPixelCount):
    topScore, topTruePositive, topFalsePositive = 0.0, 0.0, 0.0
    for key in classifiedDict:
        truePositivePercent = float(classifiedDict[key] / blobPixelCount)
        falsePositivePercent = 0
        if key in misClassifiedDict:
            falsePositivePercent = float(misClassifiedDict[key] / nonBlobPixelCount)
        score = truePositivePercent - falsePositivePercent
        if score > topScore:
            topScore = score
            topTruePositive = truePositivePercent
            topFalsePositive = falsePositivePercent
    
    return topScore, topTruePositive, topFalsePositive

def iterateFolders(folderName, blobOutputFile):
    currTime = str(datetime.datetime.now())
    os.mkdir("bit_mask_scores/" + currTime, 0o777)
    currPath = "bit_mask_scores/" + currTime + "/"
    blobOutputFile = currPath + blobOutputFile + ".txt"
    
    fileObj = open(blobOutputFile, 'a')
    fileObj.write("PMS parameters\t\tScore\n")
    fileObj.close()

    #Store the given similarities in a dictionary
    totalParamScore, totalMisClassifiedParamScore = {}, {}

    #All the tested PMS parameters
    for pmsParamFolder in os.listdir(folderName):
        categoryObjectScore, misClassifiedCategoryObjectScore = {}, {}
        #Iterate through the different categories in the given PMS parameter folder
        currPath = os.path.join(folderName, pmsParamFolder)
        for categoryFolder in os.listdir(currPath):
            filePath = os.path.join(currPath, categoryFolder)
            
            categoryScore, misClassifiedCategoryScore = [], []

            #Iterate through the images in the given folder
            for csvFile in os.listdir(filePath):
                pixelMisclassificationPercentage, blobPixelAccuracy = 0.0, 0.0

                # print("pmsParams: " + pmsParamFolder + "\tcategory: " + categoryFolder + "\tcsvFile: " + csvFile)
                baseImageArr, blobPixelCount, totalImagePixels = getBitArray(str(categoryFolder), csvFile[:-4] + ".jpg")
                pmsImageArr = loadCSV(folderName, pmsParamFolder, categoryFolder, csvFile)
                
                classifiedDict, misClassifiedDict = bitWiseCompare(baseImageArr, pmsImageArr)
                nonBlobPixelCount = totalImagePixels - blobPixelCount

                #Get the true positive and false positive for easy use to look at later on
                bestBlobScore, truePositive, falsePositive = calcCombinedAccuracy(classifiedDict, misClassifiedDict, blobPixelCount, nonBlobPixelCount)

                #classification score array for the category
                categoryScore.append(bestBlobScore)

            #Dictionary for the average clasification score per category
            categoryObjectScore[categoryFolder] = float(sum(categoryScore) / len(categoryScore))

        #Get the overall accuracy for the given PMS parameters
        totalParamScore[pmsParamFolder] = calcDictAccuracy(categoryObjectScore)
        # print(totalParamScore[pmsParamFolder])

        writeOutput(blobOutputFile, pmsParamFolder, str(totalParamScore[pmsParamFolder]))

    bestBlobMetric = max(totalParamScore, key=totalParamScore.get)
    
    writeOutput(blobOutputFile, "Best Metric: " + str(bestBlobMetric), "Score: " + str(totalParamScore[bestBlobMetric]))

iterateFolders("pms_csv", "pms_blob_classification_score")

