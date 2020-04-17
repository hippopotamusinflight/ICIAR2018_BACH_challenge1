# import modules
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd

from skimage.util.shape import view_as_windows
from PIL import Image


# get image and label df
image_labels_path = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018_BACH_Challenge/Photos/microscopy_ground_truth.csv"
image_label_df = pd.read_csv(image_labels_path, header=None, names=["tif","class"])

# shuffle labels to random order, reset index
# image_label_df = image_label_df.sample(frac=1, random_state=420).reset_index(drop=True)

# write translation function
def tumor_label(row):
    if row['class'] == "Normal": return 0
    if row['class'] == "Benign": return 1
    if row['class'] == "InSitu": return 2
    if row['class'] == "Invasive": return 3
    
# apply lambda function
image_label_df["tumor_categ"] = image_label_df.apply(lambda row: tumor_label(row), axis=1)


# make patches300 dir
photos_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018_BACH_Challenge/Photos/"
all_patches300_dir = os.path.join(photos_dir, "all_patches300/")

if not os.path.exists(all_patches300_dir):
    os.makedirs(all_patches300_dir)
else:
    print("directory exists.")
    
    
# GET PATCHES
all_images_path = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018_BACH_Challenge/Photos/all/"

patches_label_df = pd.DataFrame()

for index, row in image_label_df.iterrows():
    # print(row['tif'], row['class'])

# index, row = next(image_label_df.iterrows())
    
    img_BGR = cv2.imread(os.path.join(all_images_path, row['tif']))

    window_shape = (300, 300)
    step=299
    img_patches_B = view_as_windows(img_BGR[:,:,0], window_shape, step=step)
    img_patches_G = view_as_windows(img_BGR[:,:,1], window_shape, step=step)
    img_patches_R = view_as_windows(img_BGR[:,:,2], window_shape, step=step)

    # get bkgd and non-bkgd patches
    # all_patches = []
    non_bkgd_patches = []
    bkgd_patches = []

    for i in range(img_patches_B.shape[0]):
        for j in range(img_patches_B.shape[1]):
            img_patch = np.dstack((img_patches_B[i,j], img_patches_G[i,j], img_patches_R[i,j]))

            thresh = 200
            img_patch_thresh = cv2.threshold(img_patch, thresh, 255, cv2.THRESH_BINARY)[1]

            white_thresh = 254
            nwhite_total = img_patch_thresh.shape[0]*img_patch_thresh.shape[1]
            nwhite = 0

            for p in range(img_patch_thresh.shape[0]):
                for q in range(img_patch_thresh.shape[1]):
                    if all(pixel > white_thresh for pixel in list(img_patch_thresh[p,q,:])):
                        nwhite += 1
    #         all_patches.append(img_patch)

            bkgd_thresh = 0.7
            if nwhite/nwhite_total < bkgd_thresh:
                non_bkgd_patches.append(img_patch)
            else:
                bkgd_patches.append(img_patch)

    for i, patch in enumerate(non_bkgd_patches):
        im = Image.fromarray(patch)
        f_name = all_patches300_dir + os.path.splitext(row['tif'])[0] + "_" + str(i).zfill(3) + ".tif"
        im.save(f_name)
        patches_label_df = patches_label_df.append({'filename': (os.path.splitext(row['tif'])[0] + "_" + str(i).zfill(3) + ".tif"), "tumor_categ": row['tumor_categ']}, ignore_index=True)
    
    print(row['tif'], "patch extraction done.")


patches_label_df['tumor_categ'] = patches_label_df['tumor_categ'].astype(int)
print(patches_label_df.shape)


# write csv
all_patches300_dir = os.path.join(photos_dir, "all_patches300/")
df_name = "patches_and_labels_300_stride300.csv"

patches_label_df.to_csv(os.path.join(all_patches300_dir, df_name), sep='\t')

# EOF