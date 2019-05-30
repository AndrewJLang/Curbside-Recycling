import cv2
import numpy as np

# File that will be the focus point for now
# Finding the contours of each frame to detect the moving object and hopefully project the objects trajectory

capture = cv2.VideoCapture("blue_background_sample_images/slow_motion/IMG_4212_Slomo.mp4")

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

if capture.isOpened():
    success, frame = capture.read()
else:
    print("Video not opened successfully")
    success = False

success, frame1 = capture.read()
success, frame2 = capture.read()

while success:
    success, frame = capture.read()

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    success, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((3,3),np.uint8),iterations=3)

    #thresh gives the best results as it doesn't take into account background like dilated and blur do
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
    for contour in contours:
        M = cv2.moments(contour)
        cX, cY = 0, 0
        if M["m00"] != 0:
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
        
        cv2.circle(frame1, (cX, cY), 7, (255,255,255), -1)

    frame1 = rescale_frame(frame1, percent=35)
    cv2.imshow("inter", frame1)

    if cv2.waitKey(40) == 27:
        break

    frame1 = frame2
    ret, frame2 = capture.read()

cv2.destroyAllWindows()
cap.release(s)