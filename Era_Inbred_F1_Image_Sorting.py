#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:19:35 2022

@author: michael
"""
### Era Inbred and F1 Sorting
### Michael Burns
### 3/14/2022
### Kernel Image Sorting
### Michael Burns
### 1/26/22

# Purpose: To sort images into folders for different people to work on.
# ~100 kernels should be shared across people (this number may increase)
# The rest of the images should be evenly assigned to different people.

###################
# Import Packages #
###################
from glob import glob
from skimage import io
from tqdm import tqdm

######################
# Get List of Images #
######################
images = sorted(glob('Data/Images/Annotation_Images/Era_Hybrid_Images/Split_Era_Images/*.tif'))

print(images)
print(len(images))

folder_count = 1

for image in tqdm(images):
   kernel = io.imread(image, plugin = 'pil')
   fname = image.split('/')[-1]
   io.imsave('Data/Images/Annotation_Images/Era_Hybrid_Images/Era_Images_' + str(folder_count) + 'of6/' + fname, kernel)
   if folder_count < 6:
       folder_count += 1
   else:
       folder_count = 1