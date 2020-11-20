from PIL import Image
import os
import numpy as np
import sys
import csv

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
    countBlack, countWhite = 0,0

    for x in range(bitImage.size[1]):
        for y in range(bitImage.size[0]):
            rgbVal = int(sum(list(pixels[y,x])))
            bitVal = pixelSums[min(range(len(pixelSums)), key = lambda i: abs(pixelSums[i] - rgbVal))]
            if bitVal == 0:
                arr[x][y] = 1
            else:
                arr[x][y] = 0

    return arr


#Working properly
def loadCSV(folderName, pmsParameters, categoryName, fileName):
    pmsArr = []
    dataFile = open(folderName + "/" + pmsParameters + "/" + categoryName + "/" + fileName, 'r')
    reader = csv.reader(dataFile, delimiter=",")
    for row in reader:
        intRow = [int(i) for i in row]
        pmsArr.append(intRow)
    
    return np.array(pmsArr)

#loadCSV("pms_csv", "sr10rr10md100", "cardboard","image00018.csv")

def bitWiseCompare(baseImage, pmsImage):
    assert len(baseImage) == len(pmsImage)
    assert len(baseImage[0]) == len(pmsImage[0])

    similarityDict = {}

    for x in range(len(baseImage)):
        for i in range(len(baseImage[0])):
            if baseImage[x][i] == 0:
                continue
            else:
                if pmsImage[x][i] in similarityDict:
                    similarityDict[pmsImage[x][i]] += 1
                else:
                    similarityDict[pmsImage[x][i]] = 1
    
    return similarityDict



def iterateFolders(folderName):
    #Store the given similarities in a dictionary
    totalParamScore = []

    #All the tested PMS parameters
    for pmsParamFolder in os.listdir(folderName):
        pmsParamScore = []
        #Iterate through the different categories in the given PMS parameter folder
        currPath = os.path.join(folderName, pmsParamFolder)
        for categoryFolder in os.listdir(currPath):
            filePath = os.path.join(currPath, categoryFolder)
            categoryScore = []
            #Iterate through the images in the given folder
            for csvFile in os.listdir(filePath):
                baseImageArr = getBitArray(str(categoryFolder), csvFile[:-4] + ".jpg")
                pmsImageArr = loadCSV(folderName, pmsParamFolder, categoryFolder, csvFile)
                
                print(bitWiseCompare(baseImageArr, pmsImageArr))



iterateFolders("pms_csv")



