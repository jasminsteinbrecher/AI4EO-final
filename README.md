# AI4EO-final
Training various models on the Power Plant Satellite Imagery Dataset.

## Imports 
The following code lists all the required libraries and imports necessary to run the code:

```
import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2d, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.preprocessing import normalize
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras_unet_collection import models, losses
from tensorflow.keras.optimizers import Adam
from keras_unet.models import custom_unet
import pickle
```
