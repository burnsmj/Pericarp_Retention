#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:31:06 2024

@author: michael
"""
### Kernel Shape Extraction
### Michael Burns
### 2024/08/28

# Purpose: This script is meant to read in images of kernels that were collected 
#          before the cooking process in order to extract various shape features
#          of the kernels that could be correlated with the pericarp retention of
#          the kernels after cooking. The script will read in the images, convert
#          them to grayscale, threshold them, and then extract the shape features
#          of the kernels. The features that will be extracted are the area, perimeter,
#          major axis length, minor axis length, eccentricity, and solidity of the kernels.
#          These features will be saved to a csv file for further analysis in R.

# Import packages
from skimage import io
from skimage import measure
from skimage import color
from skimage import morphology
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# List out image file paths
image_files = glob.glob("/Users/michael/Desktop/Grad_School/Research/Pericarp/Chemometrics/Shape_Images/*.tif")

# Create data storage variable
storage_df = pd.DataFrame()

# Iterate through each image in dataset
count = 0
for image in sorted(image_files):
    # Extract sample information (naming convention is genotype_hotplate.tif)
    image_info = image.split('/')
    filename = image_info[-1]
    file_info = filename.split('.')
    sample_id = file_info[0]
    #hotplate = file_info[1].split('.')[0]
    
    # Display current iter information
    #print('---', sample_id, hotplate, '---')
    
    # Read in image
    #print('--- reading image ---')
    img_rgb = io.imread(image, plugin = 'pil')
    
    img_rgb_reduced = transform.resize(img_rgb, (1404,1020)) # make everything 120ppi - the smallest image size. Not sure why some are so small, but I remember having to reset a setting in vuescan related to scaling?
    
    # Convert image to grayscale
    #print('--- converting image ---')
    img_gry = color.rgb2gray(img_rgb_reduced)
    
    # Threshold images
    #print('--- masking image ---')
    img_msk = np.zeros_like(img_gry, dtype='bool')
    img_msk[img_gry > 0.3] = 1
    #img_msk[0:3500, :] = 0 # remove top portion of images - some have a box
    #img_msk[-700:-1, :] = 0 # remove bottom portion of images - some have a box
    #img_msk[:, 0:850] = 0 # remove left portion of images - some have a box
    #img_msk[:, -750:-1] = 0 # remove right portion of images - some have a box
    
    # Remove small objects
    #print('--- removing small objects ---')
    img_msk = morphology.remove_small_objects(img_msk, 500)
    
    # Close holes
    #print('--- closing holes ---')
    img_msk = morphology.remove_small_holes(img_msk, 500)
    
    # Label kernels
    #print('--- labelling kernels ---')
    img_lbl = measure.label(img_msk)
    
    # Collect region props information
    #print('--- collecting kernel information ---')
    kernel_info = measure.regionprops_table(img_lbl, properties = ['label', # object ID
                                                                   'area', # number of pixels in object
                                                                   'perimeter', # number of pixels around the edge of object
                                                                   #'area_convex', # number of pixels in the smallest polygon that encloses the region
                                                                   'major_axis_length', # longest length through object
                                                                   'minor_axis_length', # longest length perpendicular to major axis
                                                                   'eccentricity', # circularity of object (0 = cirlce, 1 = ellipse)
                                                                   'extent', # rectangularity of object
                                                                   'solidity' # compactness of object
                                                                   ])
    
    kernel_info['Sample_ID'] = [sample_id] * len(kernel_info['label'])
    #kernel_info['Hotplate_ID'] = [hotplate] * len(kernel_info['label'])
    
    # Convert data to pandas table and add sample id and hotplate id
    #print('--- appending data ---')
    storage_df = storage_df.append(pd.DataFrame(kernel_info), ignore_index = True)
    
    # Display OC19 mask
    if sample_id == 'YCH23:2519':
        plt.imshow(img_msk)
        plt.axes('off')
        plt.show()
    
    # Display iteration progress
    if count % 5 == 0:
        print('Completed iteration ' + str(count + 1) + ' out of ' + str(len(image_files)))
        # Display current images for debugging
        #print('--- displaying image ---')
        print(sample_id)
        plt.imshow(img_msk, cmap = 'gray')
        plt.axis('off')
        plt.show()
    
    count += 1
    # Code to keep the scale small at the beginning
    #if count > 1:
    #    break


storage_df.to_csv("/Users/michael/Desktop/Grad_School/Research/Pericarp/Chemometrics/Data/Screening_Subset_Region_Props.csv", index = False)
