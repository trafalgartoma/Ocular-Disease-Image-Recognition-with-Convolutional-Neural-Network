This repository contains the majority of the files used during the development of the thesis project. 
It was mostly used as a backup for the entire development, so there are many files used as tests and some development waste inside.

Below, I list the main files and their purpose.
Data cleaning and image pre-processing:
- odir_image_crop.py  → class used for image cropping
- odir_image_crop_testing.py  →  cropping testing images
- odir_image_crop_training.py  →  cropping training images
- odir_image_resizer.py  →  class used for image resizing
- odir_image_resize_testing.py  →   resize testing images
- odir_image_resize_training.py  →  resize training images
- ODIR_labelling_operations.py  → reworking of dataset labels, elimination of corrupted images.
- ODIR_Distribuzione_Malattie.py  → check single-label distribution
- ODIR_Distribuzione_Malattie2.py  → check multi-label distribution (quantity)
- ODIR_Model_ResNet50.py  →  model training on ResNet50
- ODIR_Model_InceptionV3.py  →  model training on InceptionV3
- ODIR_Model_VGG16.py  →  model training on VGG16
- EvaluateModel_ResNet50.py  →  evaluate model score ResNet50
- EvaluateModel_InceptionV3.py  →  evaluate model score InceptionV3
- EvaluateModel_VGG16.py   →  evaluate model score VGG16
