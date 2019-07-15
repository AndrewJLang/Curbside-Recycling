import numpy as np
import librosa as lb
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import spatial, signal

fileDirectoryPlasticBottles = "../blue_background_sample_images/trimmed_Slomo_audio/plastic_bottles"
audioFilesPlasticBottles = glob(fileDirectoryPlasticBottles + "/*.wav")

fileDirectorySodaCans = "../blue_background_sample_images/trimmed_Slomo_audio/soda_cans"
audioFilesSodaCans = glob(fileDirectorySodaCans + "/*.wav")

fileDirectoryTennisBalls = "../blue_background_sample_images/trimmed_Slomo_audio/tennis_balls"
audioFilesTennisBalls = glob(fileDirectoryTennisBalls + "/*.wav")

def extractFFT(audioArr):
    y, sr = lb.load(audioArr)
    print(f"y: {len(y)}")
    fourier = np.fft.fft(y)
    return fourier

dist1 = extractFFT(audioFilesSodaCans[0])
dist2 = extractFFT(audioFilesSodaCans[0])

if len(dist1) > len(dist2):
    dist1 = dist1[:len(dist2)]
else:
    dist2 = dist2[:len(dist1)]

print(f"dist1: {len(dist1)}\tdist2: {len(dist2)}")

_, dist1 = signal.periodogram(dist1, return_onesided=False)
_, dist2 = signal.periodogram(dist2, return_onesided=False)

result = spatial.distance.cosine(dist1, dist2)
print(result)