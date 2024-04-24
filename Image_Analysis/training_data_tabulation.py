#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 09:15:03 2021

@author: michael
"""
### Training Data Tabulation
### Michael Burns
### 9/27/21

#def Training_Data_Tabulation(dataset, corn_color = 'white'): # Uncomment this if a function is desired.  At the time of making this, it didnt seem that important.  If you create a function with this, make sure to change the filepath that is read in during the loop below.

"""
The purpose of this script is to tabulate the training data collected in 
ImageJ.  This data comes in the form of a plain text file with section 
classifications in a header row starting with an octothorpe.  All pixel values
are pasted below the header line in R,G,B format and separated by tabs.  This 
script should use the section header as a column header, and turn each section
into a column of comma separated values.
"""

###################
# Import Packages #
###################
import pandas as pd # Data wrangling and manipulation
import numpy as np
from skimage import color

###################################################
# Create a Dictionary to Populate with Pixel Data #
###################################################
pixel_dict = {}

#######################################
# Populate Dictionary with Pixel Data #
#######################################
with open('../Data/White_3_Levels_Training_Data.txt') as imagej_data: # Load data file path to read line by line
    lines = imagej_data.readlines() # Read in dataset line by line
    for line in lines: # Loop to iterate through lines
        #print('Line:', line) # For possible future debugging; Print the line we are working on
        if line.startswith('#'): # If the line starts with a #, save it as the variable we will use to separate the data
            current_header = line.strip('#\n') # need to strip the line of whitespaces before comparing it (likely due to new line character)
        if current_header not in pixel_dict:
            pixel_dict[current_header] = []
            continue
        pixel_dict[current_header] += line.split()

###################################
# Create Empty Data Frame to Fill #
###################################
tab_data = pd.DataFrame(columns = ('Label', 'Red', 'Green', 'Blue', 
                                   'Hue', 'Saturation', 'Value', 
                                   'Luminosity', 'A_Axis', 'B_Axis'))

###################################################
# Compile Color Space Data for Training Data File #
###################################################
for name, label in pixel_dict.items():
    np_label = np.array([p.split(',') for p in label], dtype = int) # skimage color space converters work on 2D arrays

    label_rgb = np_label.astype('uint8') # Needs to by uint8 for rgb2hsv to work, trust me...
    label_hsv = color.rgb2hsv(label_rgb) 
    label_lab = color.rgb2lab(label_rgb / 255) # rgb2lab expects data to be in the range of 0-1

    concat_data = pd.DataFrame({'Label' : np.array([name] * len(label)).ravel().tolist(),
                                'Red' : label_rgb[:,0].ravel().tolist(),
                                'Green' : label_rgb[:,1].ravel().tolist(),
                                'Blue' : label_rgb[:,2].ravel().tolist(),
                                'Hue' : label_hsv[:,0].ravel().tolist(),
                                'Saturation' : label_hsv[:,1].ravel().tolist(),
                                'Value' : label_hsv[:,2].ravel().tolist(),
                                'Luminosity' : label_lab[:,0].ravel().tolist(),
                                'A_Axis' : label_lab[:,1].ravel().tolist(),
                                'B_Axis' : label_lab[:,2].ravel().tolist()},
                               columns = ('Label', 'Red', 'Green', 'Blue', 
                                                                  'Hue', 'Saturation', 'Value', 
                                                                  'Luminosity', 'A_Axis', 'B_Axis'))

    tab_data = tab_data.append(concat_data, ignore_index = True)

###############################
# Write out Tabulated Dataset #
###############################
tab_data.to_csv('../Data/White_Corn_Tabulated_3_Level_Training_Data.txt', sep = '\t', index = False)
