import librosa as lb
from glob import glob
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
from scipy import stats, signal
from scipy import interpolate

np.set_printoptions(threshold=sys.maxsize)

samplePlasticBottle = "../blue_background_sample_images/regular_trimmed_audio/plastic_bottles/trimmed_audio_0.wav"
sampleTennisBall = "../blue_background_sample_images/regular_trimmed_audio/tennis_balls/trimmed_audio_0.wav"
sampleSodaCan = "../blue_background_sample_images/regular_trimmed_audio/soda_cans/trimmed_audio_0.wav"

y_plasticBottle, sr_plasticBottle = lb.load(samplePlasticBottle)
bottle = lb.feature.melspectrogram(y=y_plasticBottle, sr=sr_plasticBottle)
y_tennisBall, sr_tennisBall = lb.load(sampleTennisBall)
ball = lb.feature.melspectrogram(y=y_tennisBall, sr=sr_tennisBall)
y_sodaCan, sr_sodaCan = lb.load(sampleSodaCan)
can = lb.feature.melspectrogram(y=y_sodaCan, sr=sr_sodaCan)

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

print(f"bottle: {np.array(bottle).shape}\nball: {np.array(ball).shape}\ncan: {np.array(can).shape}")
correlation = scaler.fit_transform(signal.correlate2d(bottle,bottle, mode='full'))
print(f"correlation: {np.array(correlation).shape}")

# print(correlation)
noCorrelation = 0
for x in range(len(correlation)):
    for i in range(len(correlation[x])):
        if correlation[x][i] < 0.1 and correlation[x][i] > -0.1:
            # print("0 correlation between the two")
            print(correlation[x][i])
            noCorrelation += 1
print(noCorrelation)

# bottle = np.array(bottle)
# ball = np.array(ball)
# can = np.array(can)

# Pearson, PearsonPVal = stats.pearsonr(can.flatten(), can.flatten())
# print(Pearson)

# Spearman, SpearmanPVal = stats.spearmanr(can, bottle)
# valueLength = int(len(Spearman[0]) / 2)

# totalSum, count = 0, 0
# for x in range(len(Spearman)):
#     for i in range(valueLength):
#         if Spearman[x][i] != 1:
#             totalSum += Spearman[x][i]
#             count += 1

# correlationMatrixCoefficient = float(totalSum / count)
# print(f"Correlation Matrix Coefficient: {correlationMatrixCoefficient}")


fig, (ax_sound1, ax_sound2, ax_corr) = plt.subplots(3,1)
# ax_sound1.plot(scaler.fit_transform(can))
ax_sound2.plot(scaler.fit_transform(bottle))
ax_corr.plot(correlation)

fig.tight_layout()
fig.show()
plt.show()