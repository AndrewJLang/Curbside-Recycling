import numpy as np
import os, sys

#This is for extracting the frames from the videos and audio clips, also creates folder for frames to be placed into

#create the file location for where the extracted videos into frames go
def createFolder(folderName):
    try:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
    except OSError:
            print("Folder already exists")


#converts the videos into frames and places them inside the associated created folder
def convertVideos(fileLocation):
    count = 1
    for x in os.listdir(fileLocation):
        file_name = os.path.splitext(x)[0]
        output_path = "/home/langa21/new_extracted_videos/video" + "_" + file_name + "_#" + str(count) + "_frames"
        createFolder(output_path)
        os.system("ffmpeg -i /home/langa21/new_extracted_videos/MockDropsDataDLSR_20200913/" + x + " -r 10 " + output_path + "/image%05d.jpg -hide_banner")
        count += 1



def convertAudio(fileLocation, folderName):
    count = 0
#     createFolder("./sample_images/" + folderName + "/audio_clips")
    createFolder("./blue_background_sample_images/regular_speed/" + folderName + "/audio_clips")
    for x in os.listdir("./" + fileLocation + "/"):

        os.system("ffmpeg -i blue_background_sample_images/regular_speed/" + folderName + "/" + x + " -f mp3 -vn blue_background_sample_images/regular_speed/" + folderName + "/audio_clips/regular_clip_" + str(count) + ".mp3 -hide_banner")
        count += 1

# Need to be changed accordingly to proper directory
convertVideos("/home/langa21/new_extracted_videos/MockDropsDataDLSR_20200913/")
# convertAudio("./blue_background_sample_images/regular_speed/tennis_balls", "tennis_balls")
