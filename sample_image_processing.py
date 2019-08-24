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
def convertVideos(fileLocation, folderName):
    count = 1
    for x in os.listdir("./" + fileLocation + "/"):
        print(x)
        createFolder("./blue_background_sample_images/mixed_video_frames/video_" + str(count) + "_frames")
        os.system("ffmpeg -i blue_background_sample_images/new_mixed_videos/" + x + " -r 4 blue_background_sample_images/mixed_video_frames/video_" + str(count) + "_frames/image%05d.jpg -hide_banner")
        count += 1

def convertAudio(fileLocation, folderName):
    count = 0
#     createFolder("./sample_images/" + folderName + "/audio_clips")
    createFolder("./blue_background_sample_images/regular_speed/" + folderName + "/audio_clips")
    for x in os.listdir("./" + fileLocation + "/"):
        # os.system("ffmpeg -i sample_images/" + folderName + "/" + x + " -f mp3 -vn sample_images/" + folderName + "/audio_clips/audio_clip_" + str(count) + ".mp3 -hide_banner")
        os.system("ffmpeg -i blue_background_sample_images/regular_speed/" + folderName + "/" + x + " -f mp3 -vn blue_background_sample_images/regular_speed/" + folderName + "/audio_clips/regular_clip_" + str(count) + ".mp3 -hide_banner")
        count += 1

# Need to be changed accordingly to proper directory
convertVideos("blue_background_sample_images/new_mixed_videos", "plastic_bottles")
# convertAudio("./blue_background_sample_images/regular_speed/tennis_balls", "tennis_balls")