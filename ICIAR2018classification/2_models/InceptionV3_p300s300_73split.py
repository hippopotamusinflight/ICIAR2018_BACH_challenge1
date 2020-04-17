# import the necessary packages

model_name = "inceptionV3_BNandDO_imgaug_p300s300_v1.h5"

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import VGG16

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout

from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import pandas as pd

########################################################################
# create the base inceptionV3 with input dim 
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(300, 300, 3)))
# base_model.summary()

# add a global spatial average pooling layer
headModel = base_model.output
headModel = GlobalAveragePooling2D()(headModel)      # add GAP to output of base model
headModel = BatchNormalization()(headModel)
headModel = Dropout(0.25)(headModel)

# add fully connected layer + softmax layer with 4 classes
headModel = Dense(1024, activation='relu')(headModel)

headModel = BatchNormalization()(headModel)
headModel = Dropout(0.5)(headModel)

# and a logistic layer -- we have 4 classes
headModel = Dense(4, activation='softmax')(headModel)

# final model to train
model = Model(inputs=base_model.input, outputs=headModel)
# model.summary()

# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    
# set up optimizer
opt = Adam(lr=1e-4)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])


########################################################################
import sys
sys.path.insert(1, "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/")

from ImageDataAugmentor.image_data_augmentor import *
from imgaug import augmenters as iaa
import imgaug as ia
# from train_val_data_labels_patches import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

print("\ngetting training and validation data and labels...\n")
data_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/1_preprocessing/"
X_train = np.load(os.path.join(data_dir,"X_train_300_byCase_70_30_split.npy"))
y_train = np.load(os.path.join(data_dir,"y_train_300_byCase_70_30_split.npy"))
X_val = np.load(os.path.join(data_dir,"X_val_300_byCase_70_30_split.npy"))
y_val = np.load(os.path.join(data_dir,"y_val_300_byCase_70_30_split.npy"))
class_num = 4


########################################################################
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


########################################################################
# train the model on the new data for a few epochs
# model.fit_generator(...)
epochs = 25
batch_size = 128

H = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,                 # 400//32 = 11
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // batch_size,
    epochs=epochs)


########################################################################
# train the top 2 inception blocks, 
# i.e. we will freeze the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:      # train conv2d_175 and onward
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again 
# (this time fine-tuning the top 2 inception blocks, with the top Dense layers)
epochs = 50
batch_size = 128

# train with fit_generator() with image augmentation
H = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,                 # 400//32 = 11
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // batch_size,
    epochs=epochs)


########################################################################
# save model and csv

outputs_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/2_pretrained_models_test/outputs/"
model.save(os.path.join(outputs_dir, model_name))

# write history to csv
import pandas as pd
hist_df = pd.DataFrame(H.history)
hist_csv_file = "./outputs/" + model_name + ".csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    
    
# EOF

