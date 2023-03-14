import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import precision_score
from tensorflow.python.client.session import InteractiveSession
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from keras.models import Model
import matplotlib.pyplot as plt
import itertools


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

models_dir = "/Users/giuse/Desktop/Models/TEST/checkpoint"

x_test = np.load('testing.npy')
y_test = np.load('testing_labels.npy')
x_train = np.load('train.npy')
y_train = np.load('train_labels.npy')
x_val = np.load('valid.npy')
y_valid = np.load('valid_labels.npy')

print("test shape: ", x_test.shape)
print("testing labels shape: ", y_test.shape)
print("train shape: ", x_train.shape)
print("train labels shape: ", y_train.shape)
print("validation shape: ", x_val.shape)
print("validation labels shape: ", y_valid.shape)

model = load_model(models_dir + '/model_InceptionV3.h5')
model.summary()

#print("Training set: ", model.evaluate(x_train, y_train))
#print("Validation set: ", model.evaluate(x_val, y_valid))

result = model.evaluate(x_test, y_test)
print(model.metrics_names)
print("Testing set: ", result)


precision = result[2]
recall = result[3]
print("precision: ",precision)
print("recall: ",recall)

f1_score = 2 * (precision * recall) / (precision + recall)
print("f1_score: ",f1_score)
'''
classes = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia']


y_true = y_test
y_pred = model.predict(x_test)

cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')
ax.set_ylim(6.5,-0.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.max() / 3.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.figure()
plt.show()
'''