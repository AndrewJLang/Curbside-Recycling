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

    return arr, countObject


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

def iterateFolders(folderName, blobOutputFile, nonblobOutputFile):
    currTime = str(datetime.datetime.now())
    os.mkdir("bit_mask_scores/" + currTime, 0o777)
    currPath = "bit_mask_scores/" + currTime + "/"
    blobOutputFile = currPath + blobOutputFile + ".txt"
    nonblobOutputFile = currPath + nonblobOutputFile + ".txt"
    
    fileObj = open(blobOutputFile, 'a')
    fileObj.write("PMS parameters\t\tScore\n")
    fileObj.close()

    fileObj = open(nonblobOutputFile, 'a')
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
                baseImageArr, blobPixelCount = getBitArray(str(categoryFolder), csvFile[:-4] + ".jpg")
                pmsImageArr = loadCSV(folderName, pmsParamFolder, categoryFolder, csvFile)
                
                classifiedDict, misClassifiedDict = bitWiseCompare(baseImageArr, pmsImageArr)
                mainClassification = max(classifiedDict, key=classifiedDict.get)
                
                #Get accuracy at which it was able to classify just the blob
                blobPixelAccuracy = float(float(classifiedDict[mainClassification]) / float(blobPixelCount))
                # print("blob accuracy score: " + str(blobPixelAccuracy))

                #Get proportion of image that was classified as the 'main blob'
                if mainClassification in misClassifiedDict:
                    pixelMisclassificationPercentage = float(float(misClassifiedDict[mainClassification]) / float(misClassifiedDict[mainClassification] + classifiedDict[mainClassification]))
                else:
                    pixelMisclassificationPercentage = 0.0
                # print("misclassification percentage score: " + str(pixelMisclassificationPercentage))
                
                #Add correctly classified pixels score inside blob
                categoryScore.append(blobPixelAccuracy)

                #Add correctly classified pixels outside of blob
                misClassifiedCategoryScore.append(pixelMisclassificationPercentage)

            
            #Dictionary for the average clasification score per category
            categoryObjectScore[categoryFolder] = float(sum(categoryScore) / len(categoryScore))
            
            misClassifiedCategoryObjectScore[categoryFolder] = float(sum(misClassifiedCategoryScore) / len(misClassifiedCategoryScore))

        #Get the overall accuracy for the given PMS parameters
        totalParamScore[pmsParamFolder] = calcDictAccuracy(categoryObjectScore)
        print("Param accuracy: " + str(totalParamScore[pmsParamFolder]))

        totalMisClassifiedParamScore[pmsParamFolder] = calcDictAccuracy(misClassifiedCategoryObjectScore)
        print("Param misclassified percentage: " + str(totalMisClassifiedParamScore[pmsParamFolder]))

        writeOutput(blobOutputFile, pmsParamFolder, str(totalParamScore[pmsParamFolder]))
        writeOutput(nonblobOutputFile, pmsParamFolder, str(totalMisClassifiedParamScore[pmsParamFolder]))

    bestBlobMetric = max(totalParamScore, key=totalParamScore.get)
    bestNonBlobMetric = min(totalMisClassifiedParamScore, key=totalMisClassifiedParamScore.get)
    
    writeOutput(blobOutputFile, "Best Metric: " + bestBlobMetric, "Score: " + totalParamScore[bestBlobMetric])
    writeOutput(nonblobOutputFile, "Best Metric: " + bestNonBlobMetric, "Score: " + totalMisClassifiedParamScore[bestNonBlobMetric])

iterateFolders("pms_csv", "pms_blob_classification", "pms_non_blob_classification")

