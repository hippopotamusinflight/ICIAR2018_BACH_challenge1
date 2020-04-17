#!/usr/bin/env python

model_name = "CNN9layers_w_ImageDataAugmentor_imgaug_patches1400stride100_es_mc_200epochs_bs128_p100_lr4_70_30_split"

epochs=200
batch_size=128
patience=100
lr=1e-4

import os
import numpy as np
import sys
sys.path.insert(1, "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/")

from ImageDataAugmentor.image_data_augmentor import *
from imgaug import augmenters as iaa
import imgaug as ia
# from train_val_data_labels_patches import *
from keras.callbacks import ModelCheckpoint, EarlyStopping


print("\ngetting training and validation data and labels...\n")
data_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/1_preprocessing/"
X_train = np.load(os.path.join(data_dir,"X_train_1400_byCase_70_30_split.npy"))
y_train = np.load(os.path.join(data_dir,"y_train_1400_byCase_70_30_split.npy"))
X_val = np.load(os.path.join(data_dir,"X_val_1400_byCase_70_30_split.npy"))
y_val = np.load(os.path.join(data_dir,"y_val_1400_byCase_70_30_split.npy"))
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
    sometimes(iaa.ElasticTransformation(alpha=(0.8, 1.2), sigma=(9.0, 11.0))),
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


# model
print("\nconstruct and fit model\n")
from layers9model import *

model = layers9model(X_train.shape[1:], lr, class_num)

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)
mc = ModelCheckpoint((model_name + ".h5"), monitor="accuracy", mode="max", verbose=1, save_best_only=True)

H = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,                 # 11661*0.9//64 = 163 steps
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // batch_size,
    epochs=epochs,
    callbacks=[es, mc])


# write history to csv
import pandas as pd
hist_df = pd.DataFrame(H.history)
hist_csv_file = "./" + model_name + ".csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    

# saving model
# model.save("./" + model_name + ".h5")

# EOF
