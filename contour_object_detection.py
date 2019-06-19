import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import csv
import os
import time

#Create arrays to store all similar videos so they can be processed at the same time
tennisBallVideos, plasticBottleVideos, sodaCanVideos, paperVideos = [], [], [], []

#Get all the videos that are in slo-mo so we can extract all the features at once
def getAllVideos(folderName, videoArray):
    for x in os.listdir("./blue_background_sample_images/slow_motion/take2/"+ folderName + "/"):
        videoArray.append(x)
    return videoArray

getAllVideos("tennis_balls", tennisBallVideos)
getAllVideos("soda_cans", sodaCanVideos)
getAllVideos("plastic_bottles", plasticBottleVideos)
getAllVideos("paper", paperVideos)

listOfObjects = ['tennis_balls', 'soda_cans', 'plastic_bottles', 'paper']
selectedObject = 3

# capture = cv2.VideoCapture("sample_images/soda_cans/IMG_4142.MOV")

# File that will be the focus point for now
# Finding the contours of each frame to detect the moving object and hopefully project the objects trajectory
capture = cv2.VideoCapture("blue_background_sample_images/slow_motion/take2/" + listOfObjects[selectedObject] + "/" + paperVideos[8])

#Calculates the Euclidean distance between two colors to find the one that is the most similar
#Returns the position of the color in the array
def similar_color(newColorArr, prevColor=(255,255,255)):
    EuclideanDistance = 9999
    position = -1
    if len(newColorArr) == 1:
        position = 0
        return newColorArr[position], position
    for x in range(0,len(newColorArr)):
        EuclideanCurrent = (newColorArr[x][0]-prevColor[0])**2  + (newColorArr[x][1]-prevColor[1])**2 + (newColorArr[x][2]-prevColor[2])**2  
        if EuclideanCurrent < EuclideanDistance:
            EuclideanDistance = EuclideanCurrent
            position = x
    try:
        return newColorArr[position], position
    except IndexError:
        return _, position
def rescale_frame(frame, percent):
    if frame is not None:
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    else:
        return None

def GLCMImage(GLCMArray, feature=1):
    selectedFeature = []
    GLCMFeatures = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    arr = np.array([[0,0,1,1], [0,1,1,1]], dtype=np.uint8)
    # print(len(GLCMArray[0]))
    for x, patch in enumerate(GLCMArray):
        print(x)
        glcm = greycomatrix(patch, [5], [0], 10000, symmetric=True, normed=True) #Need to bin image later for optimization
        selectedFeature.append(greycoprops(glcm, prop='contrast')[0,0])
        print(selectedFeature[x])
    return selectedFeature  

def CLAHEImage(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # claheImg = clahe.apply(image.asType(np.uint8))
    # cv2.imshow("Normal Image", image)
    # cv2.imshow("Clahe image", claheImg)


#Writes the ground truths and the feature vectors to different files to run through the ANN
def writeCSV(fileName, colorArr, testObject=1, fileType='a'):
    with open("Neural Network/CSV Files/" + fileName, fileType, newline='') as f:
        fileWriter = csv.writer(f)
        for x in range(len(colorArr)):
            fileWriter.writerow(colorArr[x])
    f.close()
    with open("Neural Network/CSV Files/GroundTruths.csv", fileType, newline='') as gt:
        fileWriter = csv.writer(gt)
        for x in range(len(colorArr)):
            gt.write(str(testObject))
            gt.write("\n")
    gt.close()

if capture.isOpened():
    success, frame = capture.read()
else:
    print("Video not opened successfully")
    success = False

success, frame1 = capture.read()
success, frame2 = capture.read()

trajectory = []
lastColor, listOfColors, position = [], [], -1
GLCMArr = []

while success:
    contourColors = []
    success, frame = capture.read()
    regionArr = []
    
    if frame2 is not None:
        diff = cv2.absdiff(frame1, frame2)
    
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    success, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((3,3),np.uint8),iterations=3)

    #thresh gives the best results as it doesn't take into account background like dilated and blur do
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_area = []
    trimmed_contours = []
    trimmed_contours_area = []

    for contour in contours:
        contour_area.append(cv2.contourArea(contour))
        if cv2.contourArea(contour) > 1000:
            trimmed_contours.append(contour)
            trimmed_contours_area.append(cv2.contourArea(contour))

    for contour in trimmed_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #need to get the region of the box to extract color from
        maxX, maxY = 0,0
        minX, minY = 9999,9999
        for x in range(len(box)):
            if box[x][0] > maxX:
                maxX = box[x][0]
            if box[x][1] > maxY:
                maxY = box[x][1]
            if box[x][0] < minX:
                minX = box[x][0]
            if box[x][1] < minY:
                minY = box[x][1]

        if frame1 is None:
            break
        else:
            region = frame1[minY:maxY, minX:maxX]
            regionArr.append(region)
            b,g,r = np.mean(region, axis=(0,1))
            contourColors.append((r,g,b/2))
            cv2.drawContours(frame1, [box], 0, (0,0,255), 2)


    if len(trimmed_contours) != 0:
        if lastColor is None or len(lastColor) == 0:
            lastColor, position = similar_color(contourColors)
        else:
            lastColor, position = similar_color(contourColors, lastColor)
        listOfColors.append(lastColor)

    cv2.drawContours(frame1, trimmed_contours, -1, (0,255,0), 2)


    count = 0
    for contour in trimmed_contours:
        M = cv2.moments(contour)
        cX, cY = 0, 0
        if M["m00"] != 0:
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
        
        cv2.circle(frame1, (cX, cY), 7, (255,255,255), -1)
        if count == position:
            GLCMArr.append(regionArr[position])
            trajectory.append((cX, cY))
        count += 1
    frame1 = rescale_frame(frame1, percent=35)
    if frame1 is not None:
        cv2.imshow("Contour video", frame1)
    else:
        break

    if cv2.waitKey(40) == 27:
        break

    frame1 = frame2
    ret, frame2 = capture.read()
    


#print a graph of the centroids associated with the object
def graphTrajectory(traject):
    for x in range(len(traject)):
        plt.scatter(traject[x][0], traject[x][1])
    plt.xlim(0,2000)
    plt.ylim(0,2000)

count = 0
while count < len(listOfColors):
    try:
        if listOfColors[count][2] > 100:
            listOfColors.pop(count)
    except IndexError:
        break
    count += 1

print(len(listOfColors))
formattedColorList = []
for x in range(len(listOfColors)):
    try:
        if listOfColors[x][2] > 110:
            listOfColors.pop(x)
        else:
            formattedColorList.append(listOfColors[x])
    except IndexError:
        print("Wrong item attempted to be written to file")

print(len(formattedColorList))

graphTrajectory(trajectory)
writeCSV("Dataset.csv", formattedColorList, testObject=selectedObject)

plt.show()

cv2.destroyAllWindows()
capture.release()