import numpy as np
import os, sys

#create the file location for where the extracted videos into frames go
def createFolder(folderName):
    try:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
    except OSError:
            print("Folder already exists")


#converts the videos into frames and places them inside the associated created folder
def convertVideos(fileLocation, folderName):
    count = 1
    for x in os.listdir("./" + fileLocation + "/"):        
        createFolder("./sample_images/" + folderName + "/video_" + str(count) + "_frames")
        os.system("ffmpeg -i sample_images/" + folderName + "/" + x + " -r 4 sample_images/" + folderName + "/video_" + str(count) + "_frames/image%05d.jpg -hide_banner")
        count += 1

convertVideos("./sample_images/tennis_balls", "tennis_balls")