import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

batch_size = 32
epochs = 10

data_dir = "./pms_images"

