import numpy as np
from glob import glob
from scipy import spatial
import librosa as lb
from sklearn.decomposition import PCA

plasticBottles = "../blue_background_sample_images/PCA_audio/plastic_bottles"
plasticBottles = glob(plasticBottles + "/*.wav")


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


#For extracting FFT audio
bottleFFT = np.array(extractFFT(plasticBottles))
minElement = (min(map(len, bottleFFT)))

newArr = []
for x in range(len(bottleFFT)):
    newArr.append(bottleFFT[x][:minElement])

bottleFFT = newArr

#Now perform PCA on the spectrograms of the sound clips
#This will return the values that the cosine similarity can be taken from
def pcaAnalysis(frequencyArr):
    pca = PCA(n_components=0.70, svd_solver='full')
    pca.fit(frequencyArr)
    print(f"Variance ratio: {pca.explained_variance_ratio_}\t"
        f"Values: {pca.singular_values_}")
    return pca.singular_values_

similarityArr = np.array(pcaAnalysis(bottleFFT))
similarityArr = np.reshape(similarityArr, (3,1))


result = 1 - spatial.distance.cosine(similarityArr[0], similarityArr[1])
print(f"Result: {result}")