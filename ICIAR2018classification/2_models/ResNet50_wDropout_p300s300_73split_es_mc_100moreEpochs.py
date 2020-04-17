# ResNet50_wDropout_p300s300_73split_es_mc_100moreEpochs.py

'''

'''

model_name = "ResNet50_wDropout_p300s300_73split_es_mc_100moreEpochs"

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
# FURTHER FINE TUNING MODEL

print("\nloading fine-tuned model...\n")
pretrained_models_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/2_pretrained_models_test/outputs/"
model_fname = "ResNet50_wDropout_p300s300_73split_v1.h5"
from keras.models import load_model
model = load_model(os.path.join(pretrained_models_dir, model_fname))

from keras import optimizers, losses, metrics


epochs=100
batch_size=128
patience=40

model.compile(optimizer=optimizers.Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)
mc = ModelCheckpoint(("./outputs/" + model_name + ".h5"), monitor="accuracy", mode="max", verbose=1, save_best_only=True)

H = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // batch_size,
    epochs=epochs,
    callbacks=[es, mc])


##################################################################
# save model and CSV
# outputs_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/2_pretrained_models_test/outputs/"
# model.save(os.path.join(outputs_dir, (model_name + ".h5")))

import pandas as pd
hist_df = pd.DataFrame(H.history)
hist_csv_file = "./outputs/" + model_name + ".csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)







