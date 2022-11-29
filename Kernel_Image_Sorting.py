#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:03:56 2022

@author: michael
"""
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
import pandas as pd

######################
# Get List of Images #
######################
images = sorted(glob('../Data/Images/Annotation_Images/Split_Images/*.tif'))

print(images)
print(len(images))

####################################
# Get List of Already Split Images #
####################################
p1_paths = sorted(glob('../Data/Images/Annotation_Images/Person_1/*.tif'))
p2_paths = sorted(glob('../Data/Images/Annotation_Images/Person_2/*.tif'))
p3_paths = sorted(glob('../Data/Images/Annotation_Images/Person_3/*.tif'))

###############################
# Get List of Genotype Images #
###############################
p1_files = []
p2_files = []
p3_files = []

for path in p1_paths:
    p1_files.append(path.split('/')[-1])

for path in p2_paths:
    p2_files.append(path.split('/')[-1])

for path in p3_paths:
    p3_files.append(path.split('/')[-1])

shared_files = list(set(p1_files) & set(p2_files) & set(p3_files))

############################
# Split Images Into Groups #
############################
person_1_images = images[0:502:5]
person_2_images = images[1:502:5]
person_3_images = images[3:502:5]
shared_images = images[2:502:5]
split_again = images[4:502:5]

person_1_images += split_again[0:len(split_again)+1:3]
person_2_images += split_again[1:len(split_again)+1:3]
person_3_images += split_again[2:len(split_again)+1:3]

person_1_images += shared_images
person_2_images += shared_images
person_3_images += shared_images

#######################################
# Write Images to Appropriate Folders #
#######################################
for image in tqdm(person_1_images):
    kernel = io.imread(image, plugin = 'pil')
    fname = image.split('/')[-1]

    if fname not in p1_files:
        p1_files.append(fname)
        io.imsave('../Data/Images/Annotation_Images/Person_1/' + fname, kernel)

for image in tqdm(person_2_images):
    kernel = io.imread(image, plugin = 'pil')
    fname = image.split('/')[-1]

    if fname not in p2_files:
        p2_files.append(fname)
        io.imsave('../Data/Images/Annotation_Images/Person_2/' + fname, kernel)

for image in tqdm(person_3_images):
    kernel = io.imread(image, plugin = 'pil')
    fname = image.split('/')[-1]

    if fname not in p3_files:
        p3_files.append(fname)
        io.imsave('../Data/Images/Annotation_Images/Person_3/' + fname, kernel)

########################################
# Check Number of Files in Each Folder #
########################################
print('Number of Person 1 Images: ' + str(len(glob('../Data/Images/Annotation_Images/Person_1/*.tif'))))
print('Number of Person 2 Images: ' + str(len(glob('../Data/Images/Annotation_Images/Person_2/*.tif'))))
print('Number of Person 3 Images: ' + str(len(glob('../Data/Images/Annotation_Images/Person_3/*.tif'))))

###################################
# Confirm Number of Total Samples #
###################################
p1_set = set(p1_files)
p2_set = set(p2_files)
p3_set = set(p3_files)

p12_set = set.union(p1_set, p2_set)
p123_set = set.union(p1_set, p2_set, p3_set)

print(len(p123_set))

fnames = []
for image in images:
    fname = image.split('/')[-1]
    fnames.append(fname)

print(set(fnames) - p123_set)

##############################
# Save List of Shared Images #
##############################
#pd.DataFrame(shared_files, columns = ['File_Name']).to_csv('../Data/Images/Annotation_Images/Shared_Images_List.csv',
#                                                         index = False,
#                                                         index_label = False)
