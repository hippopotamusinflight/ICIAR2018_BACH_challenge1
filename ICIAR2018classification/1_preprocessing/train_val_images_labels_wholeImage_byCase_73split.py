from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import pandas as pd


def get_train_val_arrays(images_labels_path, all_patches_dir, resize):
    
    # get case names shuffled
    images_labels_df = pd.read_csv(images_labels_path, header=None, names=["filename","tumor_categ"])

    def tumor_label(row):
        if row['tumor_categ'] == "Normal": return 0
        if row['tumor_categ'] == "Benign": return 1
        if row['tumor_categ'] == "InSitu": return 2
        if row['tumor_categ'] == "Invasive": return 3

    # apply lambda function
    images_labels_df["tumor_categ"] = images_labels_df.apply(lambda row: tumor_label(row), axis=1)   
    # images_labels_df   
    
    images_labels_df_shuffled = images_labels_df.sample(frac=1, random_state=420).reset_index(drop=True)
    images_shuffled_list = images_labels_df_shuffled["filename"].tolist()
    labels_shuffled_list = images_labels_df_shuffled["tumor_categ"].tolist()
    images_fname_shuffled_list = [fname.split(".")[0] for fname in images_shuffled_list]

    # get train/val/test fnames list
    train_len = int(0.7*len(images_fname_shuffled_list))
    val_len = int(0.3*len(images_fname_shuffled_list))
    # test_len = int(0.2*len(images_fname_shuffled_list))

    train_images_fnames = images_fname_shuffled_list[0:train_len]
    print(len(train_images_fnames))

    val_images_fnames = images_fname_shuffled_list[train_len:train_len+val_len]
    print(len(val_images_fnames))
    

    train_image_arr_list = []
    train_label_list = []
    train_images_fnames_inloop = []
    
    val_image_arr_list = []
    val_label_list = []
    val_images_fnames_inloop = []

    
    for index, row in images_labels_df.iterrows():
        # training set
        if index == 0:
            print("starting to construct training arrays")
        for f in train_images_fnames:
            if f in row["filename"]:
                train_images_fnames_inloop.append(row["filename"])
                train_label_list.append([row["tumor_categ"]])
                img = cv2.imread(os.path.join(all_images_dir, row['filename']))
                img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_RGB_resz = cv2.resize(img_RGB, (resize, resize), interpolation=cv2.INTER_AREA)
            #         img_RGB_resz_norm = img_RGB_resz.astype('float32')
            #         img_RGB_resz_norm /= 255.0
                train_image_arr_list.append(img_RGB_resz)

        # validation set
        if index == 0:
            print("starting to construct validation arrays")        
        for f in val_images_fnames:
            if f in row["filename"]:
                val_images_fnames_inloop.append(row["filename"])
                val_label_list.append([row["tumor_categ"]])
                img = cv2.imread(os.path.join(all_images_dir, row['filename']))
                img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_RGB_resz = cv2.resize(img_RGB, (resize, resize), interpolation=cv2.INTER_AREA)
            #         img_RGB_resz_norm = img_RGB_resz.astype('float32')
            #         img_RGB_resz_norm /= 255.0
                val_image_arr_list.append(img_RGB_resz)


    X_train = np.array(train_image_arr_list)
    y_train = np.array(train_label_list)
    y_train = np_utils.to_categorical(y_train)
    class_num = y_train.shape[1]
    np.save("./X_train_wholeImage_byCase_73split", X_train)
    np.save("./y_train_wholeImage_byCase_73split", y_train)

    X_val = np.array(val_image_arr_list)
    y_val = np.array(val_label_list)
    y_val = np_utils.to_categorical(y_val)
    np.save("./X_val_wholeImage_byCase_73split", X_val)
    np.save("./y_val_wholeImage_byCase_73split", y_val)      


    zippedList = list(zip(val_images_fnames, val_label_list))
    validation_images_labels_df = pd.DataFrame(zippedList, columns = ["val_image", "val_label"])
    validation_images_labels_df.to_csv("./validation_images_labels_df_73split_wholeImage.csv", index=False)
    
    
    return X_train, y_train, X_val, y_val, class_num, train_images_fnames_inloop, val_images_fnames_inloop


if __name__ == "__main__":
    images_labels_path = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018_BACH_Challenge/Photos/microscopy_ground_truth.csv"

    photos_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018_BACH_Challenge/Photos/"
    all_images_dir = os.path.join(photos_dir, "all/")

    resize = 300
    
    get_train_val_arrays(images_labels_path, all_images_dir, resize)









