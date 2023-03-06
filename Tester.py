import numpy as np



df = np.load('training.npy')
labels = np.load('training_labels.npy')

print(df.shape)
print(labels.shape)