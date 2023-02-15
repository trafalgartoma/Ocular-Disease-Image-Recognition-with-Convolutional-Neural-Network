import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as mpimg
from tqdm import tqdm
from time import time
from collections import Counter
import random
from collections.abc import Sequence
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50, inception_v3, vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


#Codice GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
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

def generator(train_a, labels_a):
    while True:
        for i in range(len(train_a)):
            yield train_a[i].reshape(1, 250, 250, 3), labels_a[i].reshape(1, 8)

def generator_validation(test, labels):
    while True:
        for i in range(len(test)):
            yield test[i].reshape(1, 250, 250, 3), labels[i].reshape(1, 8)



models_dir = "/Users/giuse/Desktop/Models/InceptionV3"
batch_size = 16
num_classes = 8
epochs = 16


#carichiamo il dataset
df = np.load('training.npy')
labels = np.load('training_labels.npy')


train_images, val_images, train_labels, val_labels = train_test_split(df, labels, test_size=0.2)


#carichiamo il modello inception_v3
base_model = inception_v3.InceptionV3
base_model = base_model(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#model.summary()



defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=defined_metrics)

x_train = inception_v3.preprocess_input(train_images)
x_val = inception_v3.preprocess_input(val_images)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

train_datagen = Generator(x_train, train_labels, batch_size)

# With Data Augmentation
history = model.fit(generator(x_train,train_labels), steps_per_epoch=len(train_images),
                    epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[callback], validation_data=generator_validation(x_val, val_labels),
                    validation_steps=len(val_images), shuffle=False )

'''
# With Data Augmentation
history = model.fit(generator=generator(x_train,train_labels), steps_per_epoch=len(train_images),
                               epochs=epochs, verbose=1, callbacks=[callback], validation_data=generator_validation(x_val, val_labels),
                              validation_steps=len(val_images), shuffle=False )
'''

print("saving")
model.save(os.path.join(models_dir, 'model_InceptionV3.h5'))


# list all data in history
print(history.history.keys())

def plot_history(history):
    '''
    #summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    '''

    epochs = np.arange(1, len(history.history['loss']) + 1)
    print("epochs:", len(epochs))

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, train_loss, 'r-', label='train_loss')
    plt.plot(epochs, val_loss, 'g--', label='val_loss')
    plt.legend()
    print("Training and validation loss:")
    plt.show()

    train_acc = history.history['accuracy']
    val_acc = history.history['val_acc']
    plt.plot(epochs, train_acc, 'r-', label='train_acc')
    plt.plot(epochs, val_acc, 'g--', label='val_acc')
    plt.legend()
    print("Training and validation accuracy:")
    plt.show()

    lr = history.history['learning_rate']
    plt.plot(epochs, lr, 'b--', label='learning_rate')
    plt.legend()
    print("Learning rate:")
    plt.show()
plot_history(history)

# display the content of the model
baseline_results = model.evaluate(x_train, x_val, verbose=2)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()
