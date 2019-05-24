import librosa as lb
from glob import glob
import scipy
import matplotlib.pyplot as plt
import librosa.display

#Can be changed according to which folder of audio clips you want to extract from
folderName = "tennis_balls"
fileDirectoy = "sample_images/" + folderName + "/audio_clips"

audioFiles = glob(fileDirectoy + "/*.mp3") #array of all audio files

signal_arr, index_arr = [], []

#Use for cutting out where there is no noise from audio clips
def trimAudioFiles(audioArray):
    for x in range(len(audioArray)):
        y, sr = lb.load(audioArray[x])
        yr, index = lb.effects.trim(y, top_db=35)
        signal_arr.append(yr)
        index_arr.append(index)
        print(lb.get_duration(y), lb.get_duration(yr))

def graphAudio(audioArray, signalArray=None):
    if signalArray==None:
        original = plt.figure(figsize=(12, (4*9)))
        for x in range(len(audioArray)-1): #this - 1 is since it cannot graph 10 things for some reason
            y, source = lb.load(audioArray[x])
            plt.subplot(9,1,x+1)
            lb.display.waveplot(y, sr=source)
    else:
        trimmed = plt.figure(figsize=(12, (4*9)))
        for x in range(len(signalArray)-1):
            plt.subplot(9,1,x+1)
            lb.display.waveplot(signalArray[x], sr=22050) #need to make this into an integer not an array (indexArray)
    plt.show()

trimAudioFiles(audioFiles)

graphAudio(audioFiles, signalArray=signal_arr)