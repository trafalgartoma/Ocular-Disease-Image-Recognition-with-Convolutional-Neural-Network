import logging
import PIL
import os
from PIL import Image


# questa classe permette di fare un resize e il mirror delle immagini presenti nel dataset
# il mirroring viene fatto per le foto di occhi destri

class ImageResizer:

    def __init__(self, image_width, quality, source_folder, destination_folder, file_name, keep_aspect_ratio):
        self.logger = logging.getLogger('odir')
        # width desiderato per le immagini
        self.image_width = image_width
        # qualita del jpeg
        self.quality = quality
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.file_name = file_name
        self.keep_aspect_ration = keep_aspect_ratio
        logger = (
        "Elaborando il file: ", file_name, ",path source:", source_folder, ", final path:", destination_folder)
        print(logger)

    def run(self):

        # We load the original file, we resize it to a smaller width and correspondent height and
        # also mirror the image when we find a right eye image so they are all left eyes
        # carichiamo il file originale dal source folder
        file = os.path.join(self.source_folder, self.file_name)
        # si carica l'img attuale
        img = Image.open(file)

        if self.keep_aspect_ration:
            # mantiene l'aspect ration dell'originale
            width_percentage = (self.image_width / float(img.size[0]))
            height_size = int((float(img.size[1]) * float(width_percentage)))
            img = img.resize((self.image_width, height_size), PIL.Image.ANTIALIAS)
        else:
            # l'immagine viene resa quadrata
            img = img.resize((self.image_width, self.image_width), PIL.Image.ANTIALIAS)
            # l'img destra viene flippata
        if "right" in self.file_name:
            print("Right eye image found. Flipping it")
            img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(self.destination_folder, self.file_name),
                                                      optimize=True, quality=self.quality)
        else:
            img.save(os.path.join(self.destination_folder, self.file_name), optimize=True, quality=self.quality)
            print("Image saved. ")
