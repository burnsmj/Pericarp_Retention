#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:15:12 2022

@author: michael
"""
### Image Rescaling
### Michael Burns
### 10/26/22

# Purpose:  The images I collected were at 1200dpi, but to fully load the data
#           it is necessary to have a lower resolution.  This script will read
#           in an image, rescale it by a factor of 4 (1200 to 300 dpi) and then
#           write it to a new folder.  The same will be done for the image mask
#           however, we will let the computer round the pixels to the nearest
#           integer (0, 1, or 2) to keep the 3 class classification the same.

# Packages
from glob import glob
from skimage.transform import rescale
from skimage import io

# Read in file paths
images = sorted(glob('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Annotation_Images/Split_Images/*_0_nf.png'))

# Iterate through images
iter = 0
for image in images:
    # Progress Tracker
    if iter % 100 == 0:
        print('Working on iterations ' + str(iter) + ' - ' + str(iter + 99) + ' of ' + str(len(images)))
    iter += 1

    # Read in image and transform to different color spaces
    img_rgb = io.imread(image)

    # Extract the file name
    fname = image.split('/')[-1]

    # Read in the mask image
    mask = io.imread('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Annotation_Masks/' + fname, as_gray = True)
    
    # Rescale images
    img_rescale = rescale(img_rgb, 0.25, anti_aliasing = False, multichannel = True)
    mask_rescale = rescale(mask, 0.25, anti_aliasing = False, multichannel = False)
    
    # Save image
    io.imsave('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Images/' + fname[:-9] + '.png', img_rescale)
    
    # Save mask
    io.imsave('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Masks/' + fname[:-9] + '.png', mask_rescale)
