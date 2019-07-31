import numpy as np
import librosa as lb
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import spatial, signal, interpolate
from sklearn.decomposition import PCA

sameObject = False

fileDirectoryPlasticBottles = "../blue_background_sample_images/regular_trimmed_audio/plastic_bottles"
audioFilesPlasticBottles = glob(fileDirectoryPlasticBottles + "/*.wav")

fileDirectorySodaCans = "../blue_background_sample_images/regular_trimmed_audio/soda_cans"
audioFilesSodaCans = glob(fileDirectorySodaCans + "/*.wav")

fileDirectoryTennisBalls = "../blue_background_sample_images/regular_trimmed_audio/tennis_balls"
audioFilesTennisBalls = glob(fileDirectoryTennisBalls + "/*.wav")

def extractFFT(audioArr):
    y, sr = lb.load(audioArr)
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
        return 1
    elif len(dist1) < len(dist2):
        return 0
    else:
        return -1


def windowSliding(audioClip1, audioClip2):
    similarityArr = []
    if longestAudio(audioClip1, audioClip2) == 1:
        _, welch2 = signal.welch(audioClip2, return_onesided=False)
        # pcaAnalysis(np.array(welch2).reshape(1,-1))
        for x in range(0, len(audioClip1), 100):
            try:
                subClip = audioClip1[x:(len(audioClip2)+x)]
                _, welch1 = signal.welch(subClip, return_onesided=False)
                resultWelch = 1 - spatial.distance.cosine(welch1, welch2)
                similarityArr.append(resultWelch)
            except (IndexError, ValueError) as e:
                return similarityArr
    elif longestAudio(audioClip1, audioClip2) == 0:
        _, welch1 = signal.welch(audioClip1, return_onesided=False)
        # print(np.array(welch1).shape)
        for x in range(0, len(audioClip2), 100):
            try:
                subClip = audioClip2[x:(len(audioClip1)+x)]
                _, welch2 = signal.welch(subClip, return_onesided=False)
                resultWelch = 1 - spatial.distance.cosine(welch1, welch2)
                similarityArr.append(resultWelch)
            except (IndexError, ValueError) as e:
                print("Index Error or ValueError: sliding complete")
                return similarityArr
    else:
        _, welch1 = signal.welch(audioClip1, return_onesided=False)
        _, welch2 = signal.welch(audioClip2, return_onesided=False)
        resultWelch = 1 - spatial.distance.cosine(welch1, welch2)
        return resultWelch
    return similarityArr

def sameObject(audioArr1):
    maxSimilarity = []
    for x in range(len(audioArr1)):
        for i in range(len(audioArr1)):
            arr = []
            if x != i and x < i:
                try:
                    arr = windowSliding(audioArr1[x], audioArr1[i])
                    arr = np.array(arr)
                    maxSimilarity.append(max(arr))
                except TypeError:
                    maxSimilarity.append(arr)
                    continue
    return maxSimilarity


def allVideoSliding(audioArr1, audioArr2):
    maxSimilarity = []
    for x in range(len(audioArr1)):
        for i in range(len(audioArr2)):
            arr = []
            # if audioArr1[x] != audioArr2[i]:
                # print(f"Clip 1 shape: {audioArr1[x].shape}\tClip 2 shape: {audioArr2[i].shape}")
            try:
                arr = (windowSliding(audioArr1[x], audioArr2[i]))
                arr = np.array(arr)
                # print(max(arr))
                maxSimilarity.append(max(arr))
            except TypeError:
                maxSimilarity.append(arr)
                continue

    return maxSimilarity

#Does a PCA analysis on the PSD of an audio clip
def pcaAnalysis(arr1, arr2):
    pca = PCA(n_components=0.95, svd_solver='full')
    pca = PCA(n_components=5)
    pca = PCA()
    combinedArr = np.concatenate(np.array(arr1), np.array(arr2))
    print(f"Combined Array: {combinedArr}\tShape: {combinedArr.shape}")
    pca.fit(combinedArr)
    print(f"Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Values: {pca.singular_values_}")

test1 = np.array([1,4,6,8,9,5,2])
test2 = np.array([3,6,1,4,6,0,7])

pcaAnalysis(test1, test2)

audioClips1 = extractAllFFT(audioFilesTennisBalls)
audioClips2 = extractAllFFT(audioFilesTennisBalls)

# combined = np.concatenate(np.array(audioClips1), np.array(audioClips2))
# print(combined.shape)

# cosineOutput = np.array(allVideoSliding(audioClips1, audioClips2)) #Use this when audioClips1 and audioClips2 are from different files

# cosineOutput = np.array(sameObject(audioClips1)) #Use this is audioClips1 and audioClips2 are extracting from the same file

# print(f"Cosine output: {cosineOutput}")

# print(cosineOutput.shape)
# for x in range(len(cosineOutput)):
#     print(cosineOutput[x])

def writeCSV(dataset1, dataset2):
    with open("CSV Files/windowSliding.csv", mode='a', newline='') as f:
        f.write(f"{dataset1} vs. {dataset2} Similarity\n")
        for x in range(len(cosineOutput)):
            f.write(f"{cosineOutput[x]}\n")
        f.write("\n")

# writeCSV("Tennis Balls", "Tennis Balls")