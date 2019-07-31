from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import librosa as lb
from glob import glob
import warnings

#Will suppress all warnings (not recommended)
warnings.filterwarnings("ignore")

plasticBottles = "../blue_background_sample_images/PCA_audio/plastic_bottles"
plasticBottles = glob(plasticBottles + "/*.wav")

sodaCans = "../blue_background_sample_images/PCA_audio/soda_cans"
sodaCans = glob(sodaCans + "/*.wav")

tennisBalls = "../blue_background_sample_images/PCA_audio/tennis_balls"
tennisBalls = glob(tennisBalls + "/*.wav")

#shuffle the arrays so they aren't in the same 'linear' order that they are in the csv file
def shuffle_arrays(arr1, arr2):
    assert len(arr1) == len(arr2)
    shuffle_arr1 = np.empty(arr1.shape, dtype=arr1.dtype)
    shuffle_arr2 = np.empty(arr2.shape, dtype=arr2.dtype)
    perm = np.random.permutation(len(arr1))
    for oldIndex, newIndex in enumerate(perm):
        shuffle_arr1[newIndex] = arr1[oldIndex]
        shuffle_arr2[newIndex] = arr2[oldIndex]
    return shuffle_arr1, shuffle_arr2

def extractFFT(audioArr, realNum=True):
    fourierArr = []
    fourierComplex = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        fourier = np.fft.fft(y)
        fourierComplex.append(fourier)
        fourier = fourier.real
        fourierArr.append(fourier)
    # print(f"Without complex: {fourierArr}")
    # print(f"With Complex: {fourierComplex}")
    if realNum is True:
        return fourierArr
    else:
        return fourierComplex

def extractSTFT(audioArr):
    STFTArr = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        stft = np.abs(lb.stft(y, n_fft=4096))
        STFTArr.append(stft[0])
    return STFTArr
        
#arrays from the FFT transform
bottleFFT = np.array(extractFFT(plasticBottles))
canFFT = np.array(extractFFT(sodaCans))
ballFFT = np.array(extractFFT(tennisBalls))

# print(f"Shape:\nBottle: {bottleFFT.shape)}\tCan: {canFFT.shape}\tBall: {ballFFT.shape}")

#arrays from the STFT transform
bottleSTFT = np.array(extractSTFT(plasticBottles))
canSTFT = np.array(extractSTFT(sodaCans))
ballSTFT = np.array(extractSTFT(tennisBalls))

labelsSTFT = []
#Assign labels according to the object (1=bottle, 2=can, 3=ball)
bottleLabels = labelsSTFT.extend(np.full(len(bottleSTFT), 1))
canLabels = labelsSTFT.extend(np.full(len(canSTFT), 2))
ballLabels = labelsSTFT.extend(np.full(len(ballSTFT), 3))


STFTArr = np.append(bottleSTFT, canSTFT, axis=0)
STFTArr = np.append(STFTArr, ballSTFT, axis=0)

labelsSTFT, valuesSTFT = shuffle_arrays(np.array(labelsSTFT), np.array(STFTArr))

# print(f"shape: {np.array(extractSTFT(plasticBottles)).shape}")
# print(f"STFT: {extractSTFT(plasticBottles)}")

def LDA(frequencyArr, labels):
    splitMark = int(len(frequencyArr)*0.8)
    trainingData = frequencyArr[:splitMark]
    validationData = frequencyArr[splitMark:]

    lda = LinearDiscriminantAnalysis()
    lda.fit(trainingData,labels[:splitMark])
    
    validationLabels = labels[splitMark:]
    prediction = lda.predict(validationData)

    assert len(validationLabels) == len(prediction)

    count = 0
    for x in range(len(validationLabels)):
        if validationLabels[x] == prediction[x]:
            count += 1
    accuracy = float(count / len(prediction))

    print(f"training Length: {len(trainingData)}\tvalidation length: {len(validationData)}")
    print(f"validation labels: {validationLabels}\tpredictions: {prediction}")
    print(f"Accuracy: {accuracy}")

#LDA on STFT
LDA(STFTArr, labelsSTFT)

