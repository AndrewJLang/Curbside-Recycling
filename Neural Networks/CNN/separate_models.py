import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
from keras_preprocessing import image
import matplotlib.pyplot as plt
import os
import csv
import argparse

#python modules
import constants.py

"""
This class is for creating individual models to train/run on
There will be 5 models created and 5 SEPARATE NN trained
"""

#Ball model
modelBall = Sequential()
modelBall.add(Conv2D)