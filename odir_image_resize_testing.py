from os import listdir
from os.path import isfile, join
from odir_image_resizer import ImageResizer

source_path = r'/Users/giuse/Desktop/ML_in_Health_Applications/resized_data/testing_data_cropped'
destination_path = r'/Users/giuse/Desktop/ML_in_Health_Applications/resized_data/testing_data_resized'

#creiamo la funziona che processa tutte le immagini

def process_all_images():
    files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
    for file in files:
        ImageResizer(image_width, quality, source_folder, destination_folder, file, keep_aspect_ratio).run()

if __name__ == '__main__':
    image_width = 250
    keep_aspect_ratio = False
    # imposta la qualita del jpeg risultante al 100%
    quality = 100
    #path dove sono presenti i file
    source_folder = source_path
    destination_folder = destination_path
    #si chiama la funzione per processare i file
    process_all_images()
    print("PROCESSO TERMINATO")

