import numpy as np
from glob import glob
from scipy import spatial
import librosa as lb
from sklearn.decomposition import PCA

plasticBottles = "../blue_background_sample_images/PCA_audio/plastic_bottles"
plasticBottles = glob(plasticBottles + "/*.wav")

sodaCans = "../blue_background_sample_images/PCA_audio/soda_cans"
sodaCans = glob(sodaCans + "/*.wav")

tennisBalls = "../blue_background_sample_images/PCA_audio/tennis_balls"
tennisBalls = glob(tennisBalls + "/*.wav")

#This will convert it from wavelength (y-axis) to frequency (y-axis), x-axis is time
#Uses the FFT transformation and only returns the real numbers (complex removed)
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

#Extracting bottle FFT
bottleFFT = np.array(extractFFT(plasticBottles))
minElement = (min(map(len, bottleFFT)))

newArr = []
for x in range(len(bottleFFT)):
    newArr.append(bottleFFT[x][:minElement])

bottleFFT = newArr
# print(np.array(bottleFFT).shape)

#Extracting can FFT
canFFT = np.array(extractFFT(sodaCans))
minElement = (min(map(len, canFFT)))

newArr = []
for x in range(len(canFFT)):
    newArr.append(canFFT[x][:minElement])

canFFT = newArr

#Extracting ball FFT
ballFFT = np.array(extractFFT(tennisBalls))
minElement = (min(map(len, ballFFT)))

newArr = []
for x in range(len(ballFFT)):
    newArr.append(ballFFT[x][:minElement])

ballFFT = newArr

#Now perform PCA on the spectrograms of the sound clips
#This will return the values that the cosine similarity can be taken from
def pcaAnalysis(frequencyArr):
    pca = PCA(n_components=0.90, svd_solver='full')
    pca.fit(frequencyArr)
    print(f"Variance ratio: {pca.explained_variance_ratio_}\tNumber of elements: "
    f"{len(pca.explained_variance_ratio_)}")
    pca_data = pca.transform(frequencyArr)
    return pca_data

bottlePCA = np.array(pcaAnalysis(bottleFFT))
print(bottlePCA.shape)
print(f"Bottle transform data: {bottlePCA}")

canPCA = np.array(pcaAnalysis(canFFT))
print(canPCA.shape)
print(f"Bottle transform data: {canPCA}")

ballPCA = np.array(pcaAnalysis(ballFFT))
print(ballPCA.shape)
print(f"Bottle transform data: {ballPCA}")

#This should return a value between -1 and 1 I believe, with 0 being they are the exact same audio
result = 1 - spatial.distance.cosine(ballPCA[4], ballPCA[2])
print(f"Result: {result}")