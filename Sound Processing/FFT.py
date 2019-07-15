import numpy as np
import librosa as lb
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import spatial, signal, interpolate

fileDirectoryPlasticBottles = "../blue_background_sample_images/regular_trimmed_audio/plastic_bottles"
audioFilesPlasticBottles = glob(fileDirectoryPlasticBottles + "/*.wav")

fileDirectorySodaCans = "../blue_background_sample_images/regular_trimmed_audio/soda_cans"
audioFilesSodaCans = glob(fileDirectorySodaCans + "/*.wav")

fileDirectoryTennisBalls = "../blue_background_sample_images/regular_trimmed_audio/tennis_balls"
audioFilesTennisBalls = glob(fileDirectoryTennisBalls + "/*.wav")

def extractFFT(audioArr):
    y, sr = lb.load(audioArr)
    print(f"y: {len(y)}")
    fourier = np.fft.fft(y)
    return fourier

def extractAllFFT(audioArr):
    fourierArr = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        fourier = np.fft.fft(y)
        fourierArr.append(fourier)
    return fourierArr

def longestAudio(dist1, dist2):
    if len(dist1) > len(dist2):
        dist1 = dist1[:len(dist2)]
    else:
        dist2 = dist2[:len(dist1)]
    return dist1, dist2




def groupSimilarity():
    bottles = extractAllFFT(audioFilesPlasticBottles)
    cans = extractAllFFT(audioFilesSodaCans)
    balls = extractAllFFT(audioFilesTennisBalls)
    similarity = []
    count = 0
    for x in range(len(bottles)):
        for i in range(len(cans)):
            welchLen1, welchLen2 = longestAudio(bottles[x], cans[i])
            _, welchLen1 = signal.welch(welchLen1, return_onesided=False)
            _, welchLen2 = signal.welch(welchLen2, return_onesided=False)
            result = spatial.distance.cosine(welchLen1, welchLen2)
            print(result)
            similarity.append(result)
    return float(sum(similarity)/len(similarity))

print(f"Group similarity: {groupSimilarity()}")

#To test individual audio clips against each other
clip1 = extractFFT(audioFilesSodaCans[1])
clip2 = extractFFT(audioFilesPlasticBottles[3])


welch1, welch2 = longestAudio(clip1, clip2)
periodogram1, periodogram2 = longestAudio(clip1, clip2)

_, welch1 = signal.welch(welch1, return_onesided=False)
_, welch2 = signal.welch(welch2, return_onesided=False)

_, periodogram1 = signal.periodogram(periodogram1, return_onesided=False)
_, periodogram2 = signal.periodogram(periodogram2, return_onesided=False)

#1 similarity means they are exactly the same, 0 means no correlation
resultWelch = 1 - spatial.distance.cosine(welch1, welch2)
# print(resultWelch)

resultPeriodogram = 1 - spatial.distance.cosine(periodogram1, periodogram2)
# print(resultPeriodogram)