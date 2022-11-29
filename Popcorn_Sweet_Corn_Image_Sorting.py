#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:23:24 2022

@author: michael
"""
### Sorting Kernel Images for Popcorn and Sweet Corn
### Michael Burns
### 3/3/22

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
images = sorted(glob('Data/Images/Annotation_Images/Popcorn_Sweet_Corn_Images/Split_PS_Images/*.tif'))

#print(images)
print(len(images))

pop_genos = ('SG30A', 'I29', 'K47', 'IDS69', 'orvpop', 'fieldpop')
sweet_genos = ('IA2132', 'Ia453', 'IL778D', 'T242', 'G24S-Sh2', 'Glancaster-Sh2')
het_group = 'Unknown'

for image in images:
    kernel = io.imread(image, plugin = 'pil')
    fname = image.split('/')[-1]
    #print(fname)

    geno = fname.split('_')[0]
    #print(geno)
    
    number = int(fname.split('_')[-1].split('.')[0])
    #print(number)
    
    if geno in pop_genos:
        het_group = 'Popcorn'
    elif geno in sweet_genos:
        het_group = 'Sweet_Corn'
    else:
        print(geno)
            
    
    if number % 2 == 0:
        io.imsave('Data/Images/Annotation_Images/Popcorn_Sweet_Corn_Images/' + het_group + '_Person_1/' + fname, kernel)
    else:
        io.imsave('Data/Images/Annotation_Images/Popcorn_Sweet_Corn_Images/' + het_group + '_Person_2/' + fname, kernel)
