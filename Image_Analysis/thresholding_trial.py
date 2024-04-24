#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 09:47:51 2022

@author: michael
"""
### Thresholding Trial
### Michael Burns
### 2/8/22

# I am going to see how plausible it would be to threshold some images by a blue/yellow ratio to see if I can do this quicker.  
# This was a suggestion from Nathan, and I thought I had done it, but did not see evidence of it.
# I want to be sure that I have evidence of it.

###################
# Import Packages #
###################
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

##################
# Read in Images #
##################
image = io.imread('../Data/Images/Annotation_Images/Split_Images/P1306W-IL_20220104_1.tif')

##################
# Convert to LAB #
##################
image_lab = color.rgb2lab(image)

#####################
# Remove Background #
#####################
pixel_list = image[:,:,2].ravel()

print(pixel_list.shape)

pixel_list_lab = image_lab[:,:,2].ravel()

print(pixel_list_lab.shape)

no_background_lab = pixel_list_lab[pixel_list != 0]

print(no_background_lab.shape)

##################
# Plot Histogram #
##################
plt.hist(no_background_lab, bins = 256)
plt.show()

#######################
# Filter Image Pixels #
#######################
lab_mask = np.zeros(image_lab.shape)

lab_mask[image_lab[:,:,2] > 30] = 1
lab_mask[image_lab[:,:,2] <= 30] = 0.5
lab_mask[image[:,:,0] == 0] = 0

plt.imshow(image)
plt.imshow(lab_mask, alpha = 0.3)
plt.show()

# Thresholding seems to be very image specific.  It may be possible to find
# metrics and values that work for a given image, but it has very little
# generalization power.  Also, thresholding on the B axis has the issue that
# the germ and tip cap seem to turn yellow during cooking.
