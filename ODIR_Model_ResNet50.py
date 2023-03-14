import csv
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections.abc import Sequence
import sklearn.metrics
import tensorflow as tf
from PIL import Image, ImageFilter
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from tensorflow.keras import models, layers, optimizers, losses, metrics, utils, callbacks, applications
from keras.layers import MaxPool2D
from keras.layers import Dense, Flatten, Conv2D, Activation, Dropout
from tensorflow.keras.applications import resnet50, inception_v3, vgg16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_SIZE = 250
print(tf.__version__)

# Codice GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=3500)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# class names
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

# directory utilizzate
training_directory = r'/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/'
data_csv = "/Users/giuse/Desktop/file_labels_final.csv"  # file csv con le annotazioni
file_path = "/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/"
plot_dir = "/Users/giuse/Desktop/Models/TEST/"
models_dir = "/Users/giuse/Desktop/Models/TEST/checkpoint"
checkpoint_path = "/Users/giuse/Desktop/Models/TEST/checkpoint/"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 64
num_classes = 7
epochs = 30

# lettura del CSV
data = pd.read_csv(data_csv, sep=';', encoding='utf-8')  # lettura del csv

# print(data.head().to_string())
# print(data.columns)

###creiamo le labels###
training = []
training_labels = []


def dataAUG(original_image, labels):
    # CLAHE
    clahe_model = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = clahe_model.apply(original_image[:, :, 0])
    colorimage_g = clahe_model.apply(original_image[:, :, 1])
    colorimage_r = clahe_model.apply(original_image[:, :, 2])
    # Next we stack our equalized channels back into a single image
    img_clahe = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
    training.append(img_clahe)
    training_labels.append(labels)

    # horizontal flip
    horizontal_img = cv2.flip(original_image, 1)
    training.append(horizontal_img)
    training_labels.append(labels)

    # horizontal clahe
    colorimage_b = clahe_model.apply(horizontal_img[:, :, 0])
    colorimage_g = clahe_model.apply(horizontal_img[:, :, 1])
    colorimage_r = clahe_model.apply(horizontal_img[:, :, 2])
    img_clahe_horizontal = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
    training.append(img_clahe_horizontal)
    training_labels.append(labels)


# controlliamo le classi
normale = 0
diabete = 0

with open(data_csv) as csvDataFile:
    csv_reader = csv.reader(csvDataFile, delimiter=',')
    next(csv_reader)  # skip prima riga
    iterazioine = 0
    for row in csv_reader:
        if (iterazioine > -1):
            column_id = row[0]
            if (row[8] == '0'):  # se non è OTHER mette tutte le immagini nel dataset
                iterazioine = iterazioine + 1
                labels = [0, 0, 0, 0, 0, 0, 0]
                labels[0] = row[1]  # N
                labels[1] = row[2]  # D
                labels[2] = row[3]  # G
                labels[3] = row[4]  # C
                labels[4] = row[5]  # AMD
                labels[5] = row[6]  # HYPER
                labels[6] = row[7]  # MYO

                # carichiamo l'immagine di base
                eye_image = os.path.join(file_path, column_id)
                image = cv2.imread(eye_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_image = image.copy()

                if (labels == ['1', '0', '0', '0', '0', '0', '0']):  # controllo se è un fondo normale
                    normale = normale + 1

                    training.append(image)
                    training_labels.append(labels)
                elif labels == ['0', '1', '0', '0', '0', '0', '0']:  # fondo con diabete
                    training.append(image)
                    training_labels.append(labels)
                elif (labels != ['1', '0', '0', '0', '0', '0', '0']) and (
                        labels != ['0', '1', '0', '0', '0', '0', '0']):
                    training.append(image)
                    training_labels.append(labels)
                    dataAUG(original_image, labels)

data = np.array(training, dtype='uint8')
data_labels = np.array(training_labels, dtype='uint8')

# # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
# # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2], data.shape[3]])

print("Data Shape: ", data.shape)


def generator_train_set(train_a, labels_a):
    while True:
        for i in range(len(train_a)):
            yield train_a[i].reshape(1, 250, 250, 3), labels_a[i].reshape(1, 7)


def generator_validation_set(val_a, labels):
    while True:
        for i in range(len(val_a)):
            yield val_a[i].reshape(1, 250, 250, 3), labels[i].reshape(1, 7)


def generator_test_set(test_a, labels):
    while True:
        for i in range(len(test_a)):
            yield test_a[i].reshape(1, 250, 250, 3), labels[i].reshape(1, 7)


X_train, X_rem, y_train, y_rem = train_test_split(data, data_labels, test_size=0.2, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, shuffle=True)

print("Train: ", X_train.shape), print(y_train.shape)
print("Valid: ", X_valid.shape), print(y_valid.shape)
print("Test: ", X_test.shape), print(y_test.shape)

# Crea un'istanza dell'architettura Inception v3
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3
# Crea la base pre-tainata del modello
base_model = ResNet50
base_model = base_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)
# base_model = base_model(weights='imagenet', include_top=False)

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(num_classes, activation='sigmoid')(x)
# model = Model(inputs=base_model.input, outputs=predictions)

for i in base_model.layers:
    i.trainable = False

x = Flatten()(base_model.output)
pred = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=pred)
model.summary()

# model.summary()
with open(models_dir + 'modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# metriche di giudizio del modello.
defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]

# compilazione
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=defined_metrics)

# preprocessing  imput
x_train = resnet50.preprocess_input(X_train)
x_val = resnet50.preprocess_input(X_valid)
x_test = resnet50.preprocess_input(X_test)

# save numpy array as .npy formats
np.save('testing', x_test)
np.save('testing_labels', y_test)

# save numpy array as .npy formats
np.save('train', x_train)
np.save('train_labels', y_train)

# save numpy array as .npy formats
np.save('valid', x_val)
np.save('valid_labels', y_valid)


# Callbacks per il modello
def generate_callbacks(filepath, monitor='val_accuracy', mode='max'):
    return [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=2, verbose=0),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1),
        callbacks.ModelCheckpoint(filepath=filepath, monitor=monitor, mode=mode, save_best_only=True, save_freq='epoch',
                                  save_weights_only=True, verbose=1)
    ]


model_history = model.fit(generator_test_set(x_train, y_train),
                          steps_per_epoch=len(x_train),
                          epochs=epochs, batch_size=batch_size, verbose=1,
                          callbacks=generate_callbacks(checkpoint_path),
                          validation_data=generator_validation_set(x_val, y_valid),
                          validation_steps=len(x_val), shuffle=False)

print("...Saving Model...")
model.save(os.path.join(models_dir, 'model_ResNet50.h5'))


def plot_history(history):
    epoch = range(len(history.history['accuracy']))
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(epoch, history.history['accuracy'])
    plt.plot(epoch, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'accuracy.png'))
    plt.show()
    # summarize history for loss
    plt.plot(epoch, history.history['loss'])
    plt.plot(epoch, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'Validation Loss'], loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'loss.png'))
    plt.show()
    # summarize history for val precision
    plt.plot(epoch, history.history['val_precision'])
    plt.title('Val Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(plot_dir, 'precision.png'))
    plt.show()


plot_history(model_history)
