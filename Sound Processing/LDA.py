from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from scipy import spatial
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
        # print(f"Clip {x} sample length: {len(y)}")
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
        # print(f"STFT shape: {np.array(stft).shape}\tcomponent 0: {stft[0]}")
        STFTArr.append(stft[0])
    return STFTArr

minLengthArr = []
#arrays from the FFT transform
bottleFFT = np.array(extractFFT(plasticBottles))
minLengthArr.append(min(map(len, bottleFFT)))

canFFT = np.array(extractFFT(sodaCans))
minLengthArr.append(min(map(len, canFFT)))

ballFFT = np.array(extractFFT(tennisBalls))
minLengthArr.append(min(map(len, ballFFT)))

minElement = min(minLengthArr)

FFTArr = np.append(bottleFFT, canFFT, axis=0)
FFTArr = np.append(FFTArr, ballFFT, axis=0)

#Need to make the audio clips' FFT signals all the same length, will trim down to min value of group
newArr = []
for x in range(len(FFTArr)):
    newArr.append(FFTArr[x][:minElement])

# print(f"Shape: {np.array(newArr).shape}")
FFTArr = newArr

#Assign labels according to the object (1=bottle, 2=can, 3=ball)
labels = []
bottleLabels = labels.extend(np.full(len(bottleFFT), 1))
canLabels = labels.extend(np.full(len(canFFT), 2))
ballLabels = labels.extend(np.full(len(ballFFT), 3))


labelsFFT, valuesFFT = shuffle_arrays(np.array(labels), np.array(FFTArr))

#arrays from the STFT transform
bottleSTFT = np.array(extractSTFT(plasticBottles))
canSTFT = np.array(extractSTFT(sodaCans))
ballSTFT = np.array(extractSTFT(tennisBalls))


STFTArr = np.append(bottleSTFT, canSTFT, axis=0)
STFTArr = np.append(STFTArr, ballSTFT, axis=0)

labelsSTFT, valuesSTFT = shuffle_arrays(np.array(labels), np.array(STFTArr))

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

    # print(f"training Length: {len(trainingData)}\tvalidation length: {len(validationData)}")
    print(f"validation labels: {validationLabels}\tpredictions: {prediction}")
    print(f"Accuracy: {accuracy}")

# #LDA on FFT
# LDA(valuesFFT, labelsFFT)

#LDA on STFT
# LDA(valuesSTFT, labelsSTFT)

#NOTE: LDA does not work as well on FFT as STFT. May be lack of data; unclear exactly how STFT works

def pcaAnalysis(frequencyArr):
    pca = PCA(n_components=0.90, svd_solver='full')
    pca.fit(frequencyArr)
    print(f"Variance ratio: {pca.explained_variance_ratio_}\t"
        f"Values: {pca.singular_values_}")
    return pca.singular_values_

#NOTE: There is only 16 elements inside similarityArr instead of 21; 5 tennis balls are being left out
similarityArr = np.array(pcaAnalysis(FFTArr))
print(len(similarityArr))

bottlePCA = similarityArr[:len(bottleFFT)]
canPCA = similarityArr[len(bottleFFT):(len(canFFT)+len(bottleFFT))]
ballPCA = similarityArr[(len(bottleFFT)+len(canFFT)):]

#Check and see that they are all the proper length
# print(f"bottle: {len(bottlePCA)}\tcan: {len(canPCA)}\tball: {len(ballPCA)}")

result = 1 - spatial.distance.cosine(bottlePCA, canPCA[:5])
print(f"Bottle vs can: {result}")

# result = 1 - spatial.distance.cosine(bottlePCA, ballPCA)
# print(f"Bottle vs ball: {result}")

# result = 1 - spatial.distance.cosine(ballPCA, canPCA)
# print(f"Ball vs can: {result}")