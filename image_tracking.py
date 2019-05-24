import cv2
import sys, os
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

#takes an integer input to pick which tracking algorithm to use
# NOTE: Currently having a problem using GOTURN tracking
def createTracker(trackerType):
    if trackerType == 0:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == 1:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == 2:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == 3:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == 4:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == 5:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == 6:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == 7:
        tracker = cv2.TrackerCSRT_create()
    
    return tracker

videoTypes = ['cardboard', 'paper', 'plastic_bags', 'plastic_bottles', 'soda_cans', 'tennis_balls', 'mixed']

#select video type from videoTypes array
videoType = videoTypes[3]
videoPath = "sample_images/" + videoType + "/IMG_4150.MOV"

capture = cv2.VideoCapture(videoPath)

if not capture.isOpened():
    print("Video was not able to be opened")
    sys.exit(1)

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

success, frame = capture.read()

if not success: 
    print("Failed to read video")
    sys.exit(1)

downsizedFrame = rescale_frame(frame, percent=65)
# downsizedFrame = cv2.flip(downsizedFrame, -1)
boxes, colors = [], []
count = 0
while count < 1: #will be necessary for identifying multiple objects (need to change to loop for as many 'objects' wanting to identify)
    count += 1
    box = cv2.selectROI('MultiTracker', downsizedFrame)
    boxes.append(box)
    colors.append((randint(0,255), randint(0,255), randint(0,255)))

# print("press q to terminate, any other key to continue")
    # k = cv2.waitKey(0) & 0xFF
    # if (k == 113):
    #     break
print("Selected boxes {}".format(boxes))

trackerType = 7
multiTracker = cv2.MultiTracker_create()

for box in boxes:
    multiTracker.add(createTracker(trackerType), downsizedFrame, box)

while capture.isOpened():
    success, frame = capture.read()
    if not success:
        break
    downsizedFrame = rescale_frame(frame, percent=65)
    success, newBoxes = multiTracker.update(downsizedFrame)

    for x, newbox in enumerate(newBoxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(downsizedFrame, p1, p2, colors[x], 2, 1)
    
    cv2.imshow('MultiTracker', downsizedFrame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# downsizedFrame.release()
# cv2.destroyAllWindows()