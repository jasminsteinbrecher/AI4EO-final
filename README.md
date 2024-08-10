# AI4EO-final
This repository contains code for training various models on the Power Plant Satellite Imagery Dataset.

## Required Libraries
To run the code, you need to install the following libraries. 
You can install them using `pip` if they are not already installed:

```
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from keras_unet_collection import models, losses
from keras_unet.models import custom_unet
from matplotlib import pyplot as plt
from sklearn import metrics
from scipy import signal
from PIL import Image
import numpy as np
import json
import zipfile
import os
import rasterio
import cv2
from io import BytesIO
from tqdm import tqdm
import random
import pickle
```

## Setup Instructions
1. Install required libraries: Ensure all required libraries are installed as listed above.
2. Download the dataset
   - Repository: Download the dataset from the GitHub repository: [US Power Plant Dataset](https://github.com/bl166/USPowerPlantDataset.git)
   - Required Files: Specifically, you need to download the following:

     `uspp_landsat` directory, which contains the satellite images.

     `binary` directory, which is a subdirectory of `annotations` and contains the binary masks.
     
     Ensure that you download and extract these files to your working directory. 
4. Set up directory paths: After downloading, set up the directory paths for the dataset in the Jupyter Notebook. Update the follwoing section with your file paths:

```
## DIRECTORY SETUP

img_dir = r"your_file_path\uspp_landsat" 
mask_dir = r"your_file_path\binary" 

SIZE = 256
image_dataset = []
mask_dataset = []

# DATA EXTRACTION

```
4. Run the Code: Execute each cell in the Jupyter Notebook in sequence.
   I recommend starting with the custom CNN model for quicker feedback and verification of the setup.
   This will help ensure everything is functioning correctly before you proceed to trian themore complex pretrained U-Net model. 
