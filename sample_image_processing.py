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

def convertAudio(fileLocation, folderName):
    count = 0
    createFolder("./sample_images/" + folderName + "/audio_clips")
    for x in os.listdir("./" + fileLocation + "/"):
        os.system("ffmpeg -i sample_images/" + folderName + "/" + x + " -f mp3 -vn sample_images/" + folderName + "/audio_clips/audio_clip_" + str(count) + ".mp3 -hide_banner")
        count += 1

# Need to be changed accordingly to proper directory
# convertVideos("./sample_images/tennis_balls", "tennis_balls")
# convertAudio("./sample_images/tennis_balls", "tennis_balls")