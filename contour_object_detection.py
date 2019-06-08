import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

# File that will be the focus point for now
# Finding the contours of each frame to detect the moving object and hopefully project the objects trajectory

capture = cv2.VideoCapture("blue_background_sample_images/slow_motion/IMG_4212_Slomo.mp4") # tennis ball
# capture = cv2.VideoCapture("blue_background_sample_images/slow_motion/IMG_4225_Slomo.mp4") #plastic bottle
# capture = cv2.VideoCapture("blue_background_sample_images/slow_motion/IMG_4223_Slomo.mp4") #soda can

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
    return newColorArr[position], position

def rescale_frame(frame, percent):
    if frame is not None:
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    else:
        return None

# def GLCMImage():


if capture.isOpened():
    success, frame = capture.read()
else:
    print("Video not opened successfully")
    success = False

success, frame1 = capture.read()
success, frame2 = capture.read()

trajectory = []
lastColor, listOfColors, position = [], [], -1

while success:
    contourColors = []
    success, frame = capture.read()
    
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
        if cv2.contourArea(contour) > 200:
            trimmed_contours.append(contour)
            trimmed_contours_area.append(cv2.contourArea(contour))

    boxArr = []
    for contour in trimmed_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxArr.append(box)

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
        region = frame1[minY:maxY, minX:maxX]
        b,g,r = np.mean(region, axis=(0,1))
        # if b > g and b > r:
        #     trimmed_contours.remove(contour)
        #     continue
        # print("Red: " + str(r) + "\tGreen: " + str(g) + "\tBlue: " + str(b/2))
        
        contourColors.append((r,g,b/2))

        # print(len(contourColors))


        cv2.drawContours(frame1, [box], 0, (0,0,255), 2)

    if len(trimmed_contours) != 0:
        if lastColor is None:
            lastColor, position = similar_color(contourColors)
        else:
            lastColor, position = similar_color(contourColors, lastColor)
        listOfColors.append(lastColor)



    cv2.drawContours(frame1, trimmed_contours, -1, (0,255,0), 2)

    glcm_box = []
    count = 0
    for contour in trimmed_contours:
        M = cv2.moments(contour)
        cX, cY = 0, 0
        if M["m00"] != 0:
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
        
        cv2.circle(frame1, (cX, cY), 7, (255,255,255), -1)

        if count == position: #going in this code block multiple times sometimes
            glcm_box.append(box[position])
            trajectory.append((cX, cY))
        count += 1
    print("GLCM: " + str(len(glcm_box)))
    frame1 = rescale_frame(frame1, percent=35)
    if frame1 is not None:
        cv2.imshow("inter", frame1)
    else:
        break

    if cv2.waitKey(40) == 27:
        break

    frame1 = frame2
    ret, frame2 = capture.read()

print(trajectory)
#print a graph of the centroids associated with the object
for x in range(len(trajectory)):
    plt.scatter(trajectory[x][0], trajectory[x][1])
plt.xlim(0,2000)
plt.ylim(0,2000)
plt.show()

cv2.destroyAllWindows()
capture.release()