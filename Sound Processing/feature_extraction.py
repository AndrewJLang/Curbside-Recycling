import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
from dtw import dtw
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
import numpy as np
from scipy import spatial
from fastdtw import fastdtw

THRESHOLD = 50 #for differentiating whether the two audio clips are the same or not (>70 = not the same, 70> = same object)
signal_arr, index_arr = [], []

#method for extracting every mfcc from the directory of audio clips
def extractMFCC(audioArr):
    MFCC = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        MFCC.append(lb.feature.mfcc(y=y, sr=sr))
    return MFCC

#extracting all the chromagrams of the audio clips (my understanding is that this is not very useful)
def extractChroma(audioArr):
    chroma = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        chroma.append(lb.feature.chroma_stft(y=y, sr=sr))
    return chroma

#extracting all the spectrograms of the audio clips (Not sure how effective this methodoloy/feature vector is)
def extractSpectrogram(audioArr):
    spect = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        spect.append(lb.feature.melspectrogram(y=y, sr=sr))
    return spect

def extractSpectralFlatness(audioArr):
    spectral = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        spectral.append(lb.feature.spectral_flatness(y=y))


def MFCCSoundSimilarity(audioArr1, arr1Name, audioArr2, arr2Name, sameArr=False):
    categorizationCount, totalCount = 0, 0
    soundComparison = []
    for x in range(len(audioArr1)):
        groundArr = audioArr1[x]
        for i in range(len(audioArr2)):
            totalCount += 1
            dist, cost, acc_cost, path = dtw(groundArr.T, audioArr2[i].T, dist=euclidean)
            if dist < THRESHOLD and sameArr is True: #measures how often it is classified as same object when same object 
                categorizationCount += 1
            elif dist > THRESHOLD and sameArr is False: #measures how often it is classified as different object when different objects
                categorizationCount += 1
            soundComparison.append(dist)
            # print(f"The normalized distance between two audio clips: {dist}")
    if sameArr is True:
        categorizationCount -= len(audioArr1) #need to get rid of elements that are compared against themselves (will give 0 for distance)
        totalCount -= len(audioArr1)
    try:
        average = float(categorizationCount / totalCount)
    except ZeroDivisionError:
        print("improper file type")
    # print(f"Average Classification Accuracy: {arr1Name} vs. {arr2Name}= {average}")
    return soundComparison, average



fileDirectoryPlasticBottles = "../blue_background_sample_images/trimmed_Slomo_audio/plastic_bottles"
audioFilesPlasticBottles = glob(fileDirectoryPlasticBottles + "/*.wav")

fileDirectorySodaCans = "../blue_background_sample_images/trimmed_Slomo_audio/soda_cans"
audioFilesSodaCans = glob(fileDirectorySodaCans + "/*.wav")

fileDirectoryTennisBalls = "../blue_background_sample_images/trimmed_Slomo_audio/tennis_balls"
audioFilesTennisBalls = glob(fileDirectoryTennisBalls + "/*.wav")

y1, sr1 = lb.load(audioFilesPlasticBottles[0])
y2, sr2 = lb.load(audioFilesPlasticBottles[0])


def fastDTW(y1, sr1, y2, sr2):
    mfcc1 = lb.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = lb.feature.mfcc(y=y2, sr=sr2)
    print(f"mfcc1: {np.array(mfcc1).shape}\tmfcc2: {np.array(mfcc2).shape}")

    print(f"mfcc1: {np.array(mfcc1).shape}\tmfcc2: {np.array(mfcc2).shape}")
    dist, path = fastdtw(mfcc1, mfcc2)
    print(f"dist: {dist}\tpath: {path}")
    # similarity = spatial.distance.cosine(mfcc1, mfcc2)
    # print(f"similarity: {similarity}")

# plastic_bottles = extractMFCC(audioFilesPlasticBottles)
# tennis_balls = extractMFCC(audioFilesTennisBalls)

# test1, soda_can_vs_tennis_ball_avg = MFCCSoundSimilarity(soda_cans, "soda cans", tennis_balls, "tennis balls", sameArr=False)
# _, soda_can_avg = MFCCSoundSimilarity(soda_cans, "soda cans", soda_cans, "soda cans", sameArr=True)
# _, tennis_ball_avg = MFCCSoundSimilarity(tennis_balls, "tennis balls", tennis_balls, "tennis balls", sameArr=True)
# _, plastic_bottles_avg = MFCCSoundSimilarity(plastic_bottles, "plastic bottles", plastic_bottles, "plastic bottles", sameArr=True)
# _, plastic_bottles_vs_tennis_balls_avg = MFCCSoundSimilarity(plastic_bottles, "plastic bottles", tennis_balls, "tennis balls", sameArr=False)
# _, soda_can_vs_plastic_bottles_avg = MFCCSoundSimilarity(plastic_bottles, "plastic bottles", soda_cans, "soda cans", sameArr=False)
# print(f"Threshold: {THRESHOLD}")
# print(f"plastic bottles: {plastic_bottles_avg}\nsoda cans: {soda_can_avg}\ntennis balls: {tennis_ball_avg}\n"
    # f"plastic bottles vs. tennis balls: {plastic_bottles_vs_tennis_balls_avg}\nplastic bottles vs. soda cans: {soda_can_vs_plastic_bottles_avg}\n"
    # f"soda cans vs. tennis balls: {soda_can_vs_tennis_ball_avg}\n")
# print(test1)

# plt.show()