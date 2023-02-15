import csv
import cv2
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, utils, callbacks, applications
from keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
IMG_SIZE = 250
BATCH_SIZE = 16
CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# directory utilizzate
testing_directory = r'/Users/giuse/Desktop/ML_in_Health_Applications/final_data/testing_data/'
training_directory = r'/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/'
data_csv = "/Users/giuse/Desktop/file_labels_TEST.csv"  # file csv con le annotazioni

file_path = "/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/"

# lettura del CSV
data = pd.read_csv(data_csv, sep=';', encoding='utf-8')  # lettura del csv

print(data.head().to_string())
# print(data.columns)

###creiamo le labels###
training = []
training_labels = []

with open(data_csv) as csvDataFile:
    csv_reader = csv.reader(csvDataFile, delimiter=',')
    next(csv_reader) #skip prima riga
    iterazioine = 0
    for row in csv_reader:
        column_id = row[0]
        labels = [0, 0, 0, 0, 0, 0, 0, 0]
        labels[0] = row[1]
        labels[1] = row[2]
        labels[2] = row[3]
        labels[3] = row[4]
        labels[4] = row[5]
        labels[5] = row[6]
        labels[6] = row[7]
        labels[7] = row[8]
        iterazioine = iterazioine + 1
        print("Processing image: " ,column_id , ", di labels: " , labels)
        print("iter: ", iterazioine)
        # load# first the image from the folder
        eye_image = os.path.join(file_path, column_id)
        image = cv2.imread(eye_image)
        training.append(image)
        training_labels.append(labels)

training = np.array(training, dtype='uint8')
training_labels = np.array(training_labels, dtype='uint8')
# # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
# # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
training = np.reshape(training, [training.shape[0], training.shape[1], training.shape[2], training.shape[3]])

# save numpy array as .npy formats
np.save('training', training)
np.save('training_labels', training_labels)