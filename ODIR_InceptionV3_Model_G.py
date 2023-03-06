import csv
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, utils, callbacks, applications
from tensorflow.keras.applications import resnet50, inception_v3, vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_SIZE = 250

# Codice GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=3500)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# class names
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

# directory utilizzate
training_directory = r'/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/'
data_csv = "/Users/giuse/Desktop/file_labels_final.csv"  # file csv con le annotazioni
file_path = "/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/"
plot_dir = "/Users/giuse/Desktop/Models/InceptionV3/Glaucoma/foto"
models_dir = "/Users/giuse/Desktop/Models/InceptionV3/Glaucoma"
batch_size = 16
num_classes = 8
epochs = 16

# lettura del CSV
data = pd.read_csv(data_csv, sep=';', encoding='utf-8')  # lettura del csv

print(data.head().to_string())
# print(data.columns)

###creiamo le labels###
training = []
training_labels = []

with open(data_csv) as csvDataFile:
    csv_reader = csv.reader(csvDataFile, delimiter=',')
    next(csv_reader)  # skip prima riga
    iterazioine = 0
    for row in csv_reader:
        column_id = row[0]
        labels = [0, 0, 0, 0, 0, 0, 0, 0]
        labels[0] = row[1]  # normal
        labels[1] = row[2]  # diabetes
        labels[2] = row[3]  # glaucoma
        labels[3] = row[4]  # cataract
        labels[4] = row[5]  # amd
        labels[5] = row[6]  # hypertension
        labels[6] = row[7]  # myopia
        labels[7] = row[8]  # others
        # if(labels ==  ['1', '0', '0', '0', '0', '0', '0', '0']):
        if (labels[2] == '1'):
            print(labels)
            iterazioine = iterazioine + 1
            print("Processing image: ", column_id, ", di labels: ", labels)
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

print(training.shape)
print(training_labels.shape)


# save numpy array as .npy formats
# np.save('training', training)
# np.save('training_labels', training_labels)


# Generators
class Generator(Sequence):
    # Class is a dataset wrapper for better training performanc
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return np.math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def generator_train_set(train_a, labels_a):
    while True:
        for i in range(len(train_a)):
            yield train_a[i].reshape(1, 250, 250, 3), labels_a[i].reshape(1, 8)


def generator_validation_set(test, labels):
    while True:
        for i in range(len(test)):
            yield test[i].reshape(1, 250, 250, 3), labels[i].reshape(1, 8)


def generator_test_set(test, labels):
    while True:
        for i in range(len(test)):
            yield test[i].reshape(1, 250, 250, 3), labels[i].reshape(1, 8)


# Carichiamo il dataset appena creato
df = training
labels = training_labels

X_train, X_valid, y_train, y_valid = train_test_split(df, labels, test_size=0.2)

# Per prima cosa, dividiamo il dataset in Training set e in un dataset residuo
# X_train, X_rem, y_train, y_rem  = train_test_split(df, labels, train_size=0.8)
# Adesso vogliamo che il testing set e il validation set siano della stessa grandezza
# Andremo a dividere il dataset residuo per ottenerli.
# X_valid, X_test, y_valid, y_test  = train_test_split(X_rem, y_rem, test_size=0.5)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
# print(X_test.shape), print(y_test.shape)


# Crea un'istanza dell'architettura Inception v3
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3
# Crea la base pre-tainata del modello
base_model = inception_v3.InceptionV3
base_model = base_model(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# metriche di giudizio del modello.
defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=defined_metrics)

x_train = inception_v3.preprocess_input(X_train)
x_val = inception_v3.preprocess_input(X_valid)


# Funzione che server per valutare il modello dato, sul training, validation e testing set.
def evaluate_model(model):
    print("Training set:\tLoss: %f\tMetric: %f" % tuple(model.evaluate(x_train, y_train, verbose=0)))
    print("Validation set:\tLoss: %f\tMetric: %f" % tuple(model.evaluate(x_val, y_valid, verbose=0)))
    # print("Testing set:\tLoss: %f\tMetric: %f"% tuple(model.evaluate(x_test, y_test, verbose=0)))



# Callbacks per il modello
def generate_callbacks(filepath, monitor='val_acc', mode='max'):
    return [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=2, verbose=0), # Reduce learning rate by a factor of 10, if performance hasn't been improving for 20 epochs
        callbacks.ModelCheckpoint(filepath=filepath, monitor=monitor, mode=mode, save_best_only=True, save_freq='epoch', save_weights_only=True)
    ]
# train_datagen = Generator(x_train, y_train, batch_size)

# generator_test_set(x_test,y_test)
model.summary()
model_history = model.fit(generator_test_set(x_train, y_train),
                          steps_per_epoch=len(x_train),
                          epochs=epochs, batch_size=batch_size, verbose=1, callbacks=generate_callbacks(models_dir),
                          validation_data=generator_validation_set(x_val, y_valid),
                          validation_steps=len(x_val), shuffle=False)

print("...Saving Model...")
model.save(os.path.join(models_dir, 'model_InceptionV3_single_class.h5'))


def plot_history(history):
    epoch = range(len(history.history['accuracy']))
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(epoch, history.history['accuracy'])
    plt.plot(epoch, history.history['val_accuracy'])
    plt.title('Train and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'accuracy.png'))
    plt.show()
    # summarize history for loss
    plt.plot(epoch, history.history['loss'])
    plt.plot(epoch, history.history['val_loss'])
    plt.title('Train and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'loss.png'))
    plt.show()
    # summarize history for val precision
    plt.plot(epoch, history.history['val_precision'])
    plt.title('Val precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(plot_dir, 'precision.png'))
    plt.show()


plot_history(model_history)
