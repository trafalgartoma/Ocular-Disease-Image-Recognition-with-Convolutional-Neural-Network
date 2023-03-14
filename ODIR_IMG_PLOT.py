import os
import pandas as pd
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# directory utilizzate
training_directory = r'/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/'
data_csv = "/Users/giuse/Desktop/file_labels_final.csv"  # file csv con le annotazioni
file_path = "/Users/giuse/Desktop/ML_in_Health_Applications/final_data/training_data/"
plot_dir = "/Users/giuse/Desktop/Models/TEST/"
models_dir = "/Users/giuse/Desktop/Models/TEST/checkpoint"
checkpoint_path = "/Users/giuse/Desktop/Models/TEST/checkpoint/"
checkpoint_dir = os.path.dirname(checkpoint_path)



# lettura del CSV
data = pd.read_csv(data_csv, sep=';', encoding='utf-8')  # lettura del csv


def dataAUG(original_image, labels):
    # CLAHE
    clahe_model = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = clahe_model.apply(original_image[:, :, 0])
    colorimage_g = clahe_model.apply(original_image[:, :, 1])
    colorimage_r = clahe_model.apply(original_image[:, :, 2])
    # Next we stack our equalized channels back into a single image
    img_clahe = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)

    plt.imshow(img_clahe)
    plt.show()

    # horizontal flip
    image2 = cv2.flip(original_image, 1)


    plt.imshow(image2)
    plt.show()





with open(data_csv) as csvDataFile:
    csv_reader = csv.reader(csvDataFile, delimiter=',')
    next(csv_reader)  # skip prima riga
    iterazioine = 0
    soglia = 2000
    for row in csv_reader:
        if (iterazioine <60):
            column_id = row[0]
            if (row[8] == '0'):  # se non è OTHER mette tutte le immagini nel dataset
                labels = [0, 0, 0, 0, 0, 0, 0]
                labels[0] = row[1]  # N
                labels[1] = row[2]  # D
                labels[2] = row[3]  # G
                labels[3] = row[4]  # C
                labels[4] = row[5]  # AMD
                labels[5] = row[6]  # HYPER
                labels[6] = row[7]  # MYO
                iterazioine = iterazioine + 1

                # iterazioine = iterazioine + 1
                print("Processing image: ", column_id, ", di labels: ", labels)
                #print("iter: ", iterazioine)
                # carichiamo l'immagine di base
                eye_image = os.path.join(file_path, column_id)
                image = cv2.imread(eye_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_image = image.copy()
                if(column_id == "1_right.jpg"):
                    print("Processing image: ", column_id, ", di labels: ", labels)
                    plt.imshow(original_image)
                    plt.show()

                    dataAUG(original_image, labels)






