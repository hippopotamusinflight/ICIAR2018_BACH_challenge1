import sys
sys.path.insert(1, "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/")

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import operator
# import re

def pred_ICIAR_test(model_name):
    
    # get test images
    test_images_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018_BACH_Challenge_TestDataset/Photos/"
    test_images_fnames = os.listdir(test_images_dir)
    test_images_fnames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))


    # load model
    pred_datagen = ImageDataGenerator(rescale=1./255)

    model_dir = "/global/home/hpc4535/CISC881FinalProject/ICIAR2018classification/4_CNN9layerModel_patches/outputs/"
    model = load_model(os.path.join(model_dir, (model_name + ".h5")))


    # predict test images by patches
    all_images_path = test_images_dir

    pred_label_list = []
    pred_conf_list = []

    for case in range(len(test_images_fnames)):
    #     print("test #%s label = %d" % (case, test_labels_shuffled[case]))
    #     print("test #%s fname = %s" % (case, test_images_fnames[case]))

        img_BGR = cv2.imread(os.path.join(all_images_path, test_images_fnames[case]))
        print("%s.shape: %s" % (test_images_fnames[case], img_BGR.shape))

        window_shape = (1400, 1400)
        step=100
        img_patches_B = view_as_windows(img_BGR[:,:,0], window_shape, step=step)
        img_patches_G = view_as_windows(img_BGR[:,:,1], window_shape, step=step)
        img_patches_R = view_as_windows(img_BGR[:,:,2], window_shape, step=step)

        ################################################################
        # get bkgd and non-bkgd patches
        non_bkgd_patches = []
        bkgd_patches = []

        resize = 300
        
        for i in range(img_patches_B.shape[0]):
            for j in range(img_patches_B.shape[1]):
                img_patch = np.dstack((img_patches_B[i,j], img_patches_G[i,j], img_patches_R[i,j]))
                img_patch = cv2.resize(img_patch, (resize, resize), interpolation=cv2.INTER_AREA)

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

        non_bkgd_patches_arr = np.array(non_bkgd_patches)
        print("case", test_images_fnames[case], "non_bkgd_patches_arr.shape = ", non_bkgd_patches_arr.shape)


        ################################################################
        # make prediction of patches
        print("predicting patches for case: ", test_images_fnames[case])
        pred = model.predict(pred_datagen.flow(non_bkgd_patches_arr))

        ################################################################
        # get prediction labels
        pred_argmax = np.array([np.argmax(p) for p in pred])

        pred_dict = {}
        for i in range(4):
            pred_dict[i] = np.count_nonzero(pred_argmax == i)

        photo_wide_label = max(pred_dict.items(), key=operator.itemgetter(1))[0]
        pred_label_list.append(photo_wide_label)

        pred_conf = max(pred_dict.values())/non_bkgd_patches_arr.shape[0]
        pred_conf_list.append(pred_conf)

        print("%spredicted label = %d" % (test_images_fnames[case], photo_wide_label))
        print("%spred_confidence = %.1f"% (test_images_fnames[case], pred_conf))
        print("\n")


    # write prediction csv
    test_images_fnames_num = [int("".join(filter(str.isdigit, f))) for f in test_images_fnames]

    zippedList = list(zip(test_images_fnames_num, pred_label_list))
    prediction_df = pd.DataFrame(zippedList, columns = ["case", "class"])
    prediction_df.to_csv("./pred.csv", index=False)


if __name__ == "__main__":
    
    model_name = "CNN9layers_w_ImageDataAugmentor_imgaug_patches1400stride100_es_mc_100epochs_bs64_p100_lr4_90_10_split"
    
    pred_ICIAR_test(model_name)
    

# EOF
