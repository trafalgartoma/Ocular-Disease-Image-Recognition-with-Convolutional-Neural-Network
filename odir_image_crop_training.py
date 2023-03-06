from os import listdir
from os.path import isfile, join

# importiamo la funzione per il ridimensionamento delle immagini
from odir_image_crop import ImageCrop


# definiamo la funzione per processare tutte le img
def process_all_images():
    files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
    for file in files:
        ImageCrop(source_folder, destination_folder, file).remove_black_pixels()


if __name__ == '__main__':
    source_folder = r'/Users/giuse/Desktop/ML_in_Health_Applications/base_data/training_data_base'
    destination_folder = r'/Users/giuse/Desktop/ML_in_Health_Applications/resized_data/training_data_cropped'
    process_all_images()

print("PROCESSO TERMINATO")
