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

def extractFFT(audioArr):
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
    return fourierArr

def extractSTFT(audioArr):
    STFTArr = []
    for x in range(len(audioArr)):
        y, sr = lb.load(audioArr[x])
        stft = np.abs(lb.stft(y))
        STFTArr.append(stft[0])
    return STFTArr
        
#arrays from the FFT transform
# bottleFFT = np.array(extractFFT(plasticBottles))
# canFFT = np.array(extractFFT(sodaCans))

#arrays from the STFT transform
bottleSTFT = np.array(extractSTFT(plasticBottles))
canSTFT = np.array(extractSTFT(sodaCans))
ballSTFT = np.array(extractSTFT(tennisBalls))

wholeArr = np.append(canSTFT, bottleSTFT, axis=0)


# print(f"shape: {np.array(extractSTFT(plasticBottles)).shape}")
# print(f"STFT: {extractSTFT(plasticBottles)}")

def LDA(frequencyArr):
    splitMark = int(len(frequencyArr)*0.8)
    trainingData = frequencyArr[:splitMark]
    validationData = frequencyArr[splitMark:]
    labels = [2,1,1,1,1]

    lda = LinearDiscriminantAnalysis()
    lda.fit(trainingData,labels[:splitMark])

    print(f"prediction: {lda.predict(validationData)}")

LDA(wholeArr)