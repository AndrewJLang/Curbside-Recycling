import cv2
import numpy as np

#Attempted to implement the blobDetection built into opencv, not continuing work on this atm

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def blobDetection(folderName='tennis_balls', videoFrames='5', image='image00004.jpg'):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 50
    params.filterByCircularity = True
    params.minCircularity = 0.75
    # params.filterByConvexity = True
    # params.minConvexity = 0.87
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.7

    img = cv2.imread("sample_images/" + folderName + "/video_" + videoFrames + "_frames/" + image)
    # img = cv2.imread("blue_background_sample_images/video_" + videoFrames + "_frames/" + image)
    # newImage = rescale_frame(img, percent=45)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoint = detector.detect(img)
    keypoints = cv2.drawKeypoints(img, keypoint, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", keypoints)
    cv2.waitKey(0)

blobDetection()