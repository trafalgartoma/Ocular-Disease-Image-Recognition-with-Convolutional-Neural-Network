import os
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from collections.abc import Sequence
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50, inception_v3, vgg16
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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


plot_dir = "/Users/giuse/Desktop/Models/VGG16/VGG_foto"
models_dir = "/Users/giuse/Desktop/Models/VGG16"
batch_size = 16
num_classes = 8
epochs = 16
# Carichiamo il dataset
df = np.load('training.npy')
labels = np.load('training_labels.npy')

X_train, X_valid, y_train, y_valid = train_test_split(df, labels, test_size=0.1)

# Per prima cosa, dividiamo il dataset in Training set e in un dataset residuo
# X_train, X_rem, y_train, y_rem  = train_test_split(df, labels, train_size=0.8)
# Adesso vogliamo che il testing set e il validation set siano della stessa grandezza
# Andremo a dividere il dataset residuo per ottenerli.
# X_valid, X_test, y_valid, y_test  = train_test_split(X_rem, y_rem, test_size=0.5)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
# print(X_test.shape), print(y_test.shape)

class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

# Crea un'istanza dell'architettura Inception v3
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3
#base_model = inception_v3.InceptionV3
base_model = VGG16
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
print(model.summary())

x_train = vgg16.preprocess_input(X_train)
x_val = vgg16.preprocess_input(X_valid)


# x_test = inception_v3.preprocess_input(X_test)


# Funzione che server per valutare il modello dato, sul training, validation e testing set.
def evaluate_model(model):
    print("Training set:\tLoss: %f\tMetric: %f" % tuple(model.evaluate(x_train, y_train, verbose=0)))
    print("Validation set:\tLoss: %f\tMetric: %f" % tuple(model.evaluate(x_val, y_valid, verbose=0)))
    # print("Testing set:\tLoss: %f\tMetric: %f"% tuple(model.evaluate(x_test, y_test, verbose=0)))


# callback da chiamare in caso in caso di overfitting
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

# train_datagen = Generator(x_train, y_train, batch_size)

# generator_test_set(x_test,y_test)

model_history = model.fit(generator_test_set(x_train, y_train), steps_per_epoch=len(x_train),
                          epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[callback],
                          validation_data=generator_validation_set(x_val, y_valid),
                          validation_steps=len(x_val), shuffle=False)

print("...Saving Model...")
model.save(os.path.join(models_dir, 'model_VGG16.h5'))


def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'plot1.png'))
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(plot_dir, 'plot2.png'))
    plt.show()


plot_history(model_history)

#print("Evaluation of the model at the end of training")
#evaluate_model(model)
