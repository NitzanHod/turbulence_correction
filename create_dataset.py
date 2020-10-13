"""
Script for creating database of clean + distorted image.
written by oren, last modification: 25/03/2020
"""
import numpy as np
import cv2
from distort_module import DistortBlur
import os
from glob import glob
from utils import *

original_folder_path = os.path.normpath('C:/Users/HATAL99/PycharmProjects/model_turbulence/license plate database/')
original_images = glob(original_folder_path + '/*')
dest_folder_path_root = os.path.normpath('C:/Users/HATAL99/PycharmProjects/model_turbulence/generated_data_base/')
os.makedirs(dest_folder_path_root, exist_ok=True)

# paramater of the distortion
N = 15

for i, original_image_path in enumerate(original_images):  # running on all the images in the original database
    original_image = cv2.imread(original_image_path)

    # create a sub-directory for the the current image
    dest_folder_path = os.path.normpath(dest_folder_path_root + '/image_' + str(i))
    os.makedirs(dest_folder_path, exist_ok=True)
    # create 2 sub-sub-directories for the GT and the distorted images
    os.makedirs(dest_folder_path + '/GT', exist_ok=True)
    os.makedirs(dest_folder_path + '/distorted', exist_ok=True)
    # saving the ground_truth image  
    cv2.imwrite(dest_folder_path + '/GT/ground_truth_' + str(i) + '.jpg', original_image)

    for n_distortion in range(20):
        S = np.random.uniform(0.8, 1.4)
        sigma_blur_image = np.random.uniform(0.5, 1.5)
        distorted_image = DistortBlur(original_image,
                                      S=S,
                                      sigma_kernel_vertor_field=0.7,
                                      sigma_blur_image=sigma_blur_image,
                                      N=N,
                                      M_distortion=1000,
                                      M_blur=50)
        distorted_image = float2int(distorted_image)
        cv2.imwrite(dest_folder_path + '/distorted/distorted_' + str(n_distortion) + '.jpg', distorted_image)
