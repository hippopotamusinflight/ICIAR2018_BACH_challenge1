# ResNet50_wDropout_p300s300_73split.py

model_name = "ResNet50_wDropout_p300s300_73split_v1"

import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import pandas as pd

import sys
sys.path.insert(1, "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/")

from ImageDataAugmentor.image_data_augmentor import *
from imgaug import augmenters as iaa
import imgaug as ia


# get data
print("\ngetting training and validation data and labels...")
data_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/1_preprocessing/"
X_train = np.load(os.path.join(data_dir,"X_train_300_byCase_70_30_split.npy"))
y_train = np.load(os.path.join(data_dir,"y_train_300_byCase_70_30_split.npy"))
X_val = np.load(os.path.join(data_dir,"X_val_300_byCase_70_30_split.npy"))
y_val = np.load(os.path.join(data_dir,"y_val_300_byCase_70_30_split.npy"))
class_num = 4


# instantiate imgaug augmentation object
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

AUGMENTATIONS = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    sometimes(iaa.Affine(
        scale=(0.8, 1.2),
        rotate=(90),
        mode=ia.ALL)),
    sometimes(iaa.ElasticTransformation(alpha=(0.8, 1.2),\
                                        sigma=(9.0, 11.0))),
    sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.1))),
    sometimes(iaa.GaussianBlur((0, 0.1))),
    sometimes(iaa.MultiplyBrightness((0.65, 1.35))),
    sometimes(iaa.LinearContrast((0.5, 1.5))),
    sometimes(iaa.MultiplyHueAndSaturation((-1, 1)))
    ], random_order=True)


# instantiate datagen objects
train_datagen = ImageDataAugmentor(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
    augment = AUGMENTATIONS,
    rescale = 1./255,
    preprocess_input = None
)

val_datagen = ImageDataAugmentor(
    rescale = 1./255
)

# define the ImageNet mean subtraction (in RGB order)
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# set the mean subtraction value for each of the data augmentation objects
train_datagen.mean = mean
val_datagen.mean = mean


##################################################################
# FINE TUNING MODEL
import keras
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import models, regularizers, layers, optimizers, losses, metrics


conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(class_num, activation="softmax"))

for layer in conv_base.layers[:]:
    layer.trainable=False
    
# Compile frozen conv_base + top layers
model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model on the new data for a few epochs
epochs = 10
batch_size = 128

H = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,                 # 400//32 = 11
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // batch_size,
    epochs=epochs)

# Make last block of the conv_base trainable:
for layer in conv_base.layers[:165]:
    layer.trainable = False
for layer in conv_base.layers[165:]:
    layer.trainable = True

# Compile frozen conv_base + UNfrozen top block + top layers ... SLOW LR (1e-5)
model.compile(optimizer=optimizers.Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Long training with fine tuning
epochs = 50
batch_size = 128

H = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,                 # 400//32 = 11
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // batch_size,
    epochs=epochs)

# save model and CSV
outputs_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/2_pretrained_models_test/outputs/"
model.save(os.path.join(outputs_dir, (model_name + ".h5")))

import pandas as pd
hist_df = pd.DataFrame(H.history)
hist_csv_file = "./outputs/" + model_name + ".csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)







