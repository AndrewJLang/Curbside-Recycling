import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# img = cv2.imread("sample_images/tennis_balls/video_5_frames/image00004.jpg")
videoPath = "sample_images/tennis_balls/IMG_4134.MOV"

capture = cv2.VideoCapture(videoPath)

firstFrame = None

frame = capture.read()
# if frame is None:
#     break

frame = imutils.resize(frame, width=500)
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (21,21), 0)
if firstFrame is None:
    firstFrame = gray
    # continue    