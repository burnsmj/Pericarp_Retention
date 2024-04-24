#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:37:19 2022

@author: michael
"""
### Kernel Image Correction and Splitting
### Michael Burns
### 1/18/22

# Purpose: To correct and split images of large numbers of kernels into
#           standardized color images of individual kernels.

###################
# Import Packages #
###################
import pandas as pd
from glob import glob
import pericarp_image_analysis_functions as piaf
from skimage import io

################
# Read in Data #
################
image_paths = sorted(glob('../Data/Images/Annotation_Images/Ground_Truth_Cooked/Unsplit_GT_Images/*.tif'))
color_key = pd.read_csv('../Data/Images/Color_Correction/Spyder_Checkr_24_Color_Key.csv', delimiter = ',', skiprows = 1)

print(image_paths)
print('\n' + str(len(image_paths)) + '\n')

for image_path in image_paths:
    # Read in the image #
    image = io.imread(image_path, plugin = 'pil')
    
    # Get the file name #
    fname = image_path.split('/')[-1]
    print(fname)
    
    # Get the genotype in the image #
    geno = fname.split('_')[0]
    print(geno)
    
    # Get the date of the image #
    date = fname.split('_')[1].split('.')[0]
    print(date)

    # If the image isn't too bright already, correct it #
    if image[0,0,0] < 220:
        # Use the image date to read in a color checker #
        std_img = io.imread('../Data/Images/Color_Correction/Color_Check_' + date + '.tif', plugin = 'pil')
    
        # Correct the image color #
        corrected_image = piaf.image_color_correction(std_img, color_key, image)

        # Split the kernels into separate images #
        piaf.indiv_kernel_image_extraction(corrected_image, fpath = '../Data/Images/Annotation_Images/Ground_Truth_Cooked/Split_GT_Images/' + geno + '_' + str(date), ftype = 'tif')
    else:
        piaf.indiv_kernel_image_extraction(image, fpath = '../Data/Images/Annotation_Images/Ground_Truth_Cooked/Split_GT_Images/' + geno + '_' + str(date), ftype = 'tif')

##################
# Correct Images #
##################
#corrected_ylw = piaf.image_color_correction(std_img, image_key, ylw_corn)

#plt.subplot(1,2,1)
#plt.imshow(ylw_corn)
#plt.axis('off')
#plt.subplot(1,2,2)
#plt.imshow(corrected_ylw)
#plt.axis('off')
#plt.tight_layout()
#plt.show()

################
# Split Images #
################
#piaf.indiv_kernel_image_extraction(corrected_ylw, fname = '../Data/Images/Annotation_Images/Split_Images/ylw_corn_corrected')
