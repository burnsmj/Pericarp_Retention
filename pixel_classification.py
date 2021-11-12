#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:59:31 2021

@author: michael
"""
### Pixel Classification
### Michael Burns
### 10/27/21

# Purpose: To learn about creating predictive algorithms in python
#          To predict the classes of a pixel based on training data
#          To determin which model-parameter combo is best for my image data

# NOTE: It appears that pandas will be best for machine learning, so the first
#       function needed is to extract a kernel, followed by the tabulation of
#       its pixel values.  After this, the prediction can be done.  
# NOTE: The above pipeline is fine for quantification, but it does little for 
#       visualization.  I will need to find a way to collect pixel coordinates
#       When I tabulate the data.
# NOTE: After the model has been tested and validated, a final function that
#       aggregates all of the functions needed in the pipeline can be created.
#       Once this is done, all of the functions should be combined into one 
#       script that can uploaded to github.

####################
# Import Libraries #
####################
import numpy as np
import pandas as pd
import image_kernel_detection
import image_processing
from skimage import io
from skimage import color
from skimage import filters
from skimage import util
from skimage import measure
from skimage import morphology
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes # Base level model approach
from sklearn import svm # Complex, accurate, slow approach
from sklearn import neighbors # Clustering approach
from sklearn import tree # Decision tree approach
from sklearn import ensemble # Multimodel approach
from sklearn import neural_network # Neural network approach
import matplotlib.pyplot as plt
import matplotlib as mpl

########################
# Plotting Adjustments #
########################
mpl.rcParams['figure.dpi'] = 300 # Increasing default figure resolution

#################
# Read in Image #
#################
image_rgb = io.imread('../Data/Images/Trials/Poststain_White_MB_H2O.png')[:,:,0:3]

##############
# Blur Image #
##############
image_blur = util.img_as_ubyte(filters.gaussian(image_rgb,
                                                sigma = 5,
                                                multichannel = True,
                                                truncate = 2))

###########################
# Create Validation Image #
###########################
valid_image = image_kernel_detection.image_background_removal(image_blur)

#plt.imshow(valid_image)
#plt.axis('off')
#plt.show()

#valid_data = image_processing.tabulate_pixels(valid_image)

#########################
# Read in Training Data #
#########################
pixel_data = pd.read_table('../Data/White_Corn_Tabulated_Training_Data.txt',
                           delimiter = '\t')

print(pixel_data)

##############################
# Reformatting Training Data #
##############################
pixel_data_melt = pixel_data.melt(value_vars = ['Pericarp', 
                                                'Aleurone', 
                                                'Background'],
                                  var_name = 'Label', 
                                  value_name = 'RGB')
print(pixel_data_melt)

pixel_data_melt[['Red','Green','Blue']] = pixel_data_melt['RGB'].str.split(',', 
                                                                           expand = True)

pixel_data_melt = pixel_data_melt.iloc[:,[0,2,3,4]]

pixel_data_melt = pixel_data_melt.astype({'Label' : 'category',
                        'Red' : 'int32',
                        'Green' : 'int32',
                        'Blue' : 'int32'})

print(pixel_data_melt)

##############################################
# Function to Calculate HSV from RGB Columns #
##############################################
def hsv_from_rgb_cols(dataset, red_name = 'Red', green_name = 'Green', blue_name = 'Blue'):
    dataset['red_prime'] = dataset[red_name] / 255
    dataset['green_prime'] = dataset[green_name] / 255
    dataset['blue_prime'] = dataset[blue_name] / 255
    dataset['Cmax'] = dataset[['red_prime', 'green_prime', 'blue_prime']].max(axis = 1)
    dataset['Cmin'] = dataset[['red_prime', 'green_prime', 'blue_prime']].min(axis = 1)
    dataset['range'] = dataset['Cmax'] - dataset['Cmin']
    dataset.loc[dataset['Cmax'] == dataset['red_prime'], 'Hue'] = 60 * (((dataset['green_prime'] - dataset['blue_prime']) / dataset['range']) % 6)
    dataset.loc[dataset['Cmax'] == dataset['green_prime'], 'Hue'] = 60 * (((dataset['blue_prime'] - dataset['red_prime']) / dataset['range']) + 2)
    dataset.loc[dataset['Cmax'] == dataset['blue_prime'], 'Hue'] = 60 * (((dataset['red_prime'] - dataset['green_prime']) / dataset['range']) + 4)
    dataset.loc[np.logical_and(dataset['Cmax'] == 0, dataset['range'] == 0), 'Hue'] = 0
    dataset.loc[dataset['Cmax'] == 0, 'Saturation'] = 0
    dataset.loc[dataset['Cmax'] != 0, 'Saturation'] = dataset['range'] / dataset['Cmax']
    dataset['Value'] = dataset['Cmax']
    
    return_dataset = dataset.iloc[:, [0, 1, 2, 3, 10, 11, 12]]
    return return_dataset

full_pixel_data = hsv_from_rgb_cols(pixel_data_melt)

full_pixel_data.loc['SV_Ratio'] = full_pixel_data['Saturation'] / full_pixel_data ['Value']

###########################
# Splitting Training Data #
###########################
train, test = train_test_split(full_pixel_data,
                               test_size = 0.2, 
                               random_state = 7)

print(train)
print(test)

# NOTE: Need to plot out the training set to see what it looks like before training a model

colors = {'Pericarp':'blue', 'Aleurone':'red', 'Background':'black'}

plt.subplot(2,3,1)
plt.scatter(train['Red'], train['Green'], c = train['Label'].map(colors))
plt.xlabel('Red')
plt.ylabel('Green')
plt.subplot(2,3,2)
plt.scatter(train['Green'], train['Blue'], c = train['Label'].map(colors))
plt.xlabel('Green')
plt.ylabel('Blue')
plt.subplot(2,3,3)
plt.scatter(train['Blue'], train['Red'], c = train['Label'].map(colors))
plt.xlabel('Blue')
plt.ylabel('Red')
plt.subplot(2,3,4)
plt.scatter(train['Hue'], train['Saturation'], c = train['Label'].map(colors))
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.subplot(2,3,5)
plt.scatter(train['Saturation'], train['Value'], c = train['Label'].map(colors))
plt.xlabel('Saturation')
plt.ylabel('Value')
plt.subplot(2,3,6)
plt.scatter(train['Value'], train['Hue'], c = train['Label'].map(colors))
plt.xlabel('Value')
plt.ylabel('Hue')
plt.tight_layout()
plt.show()

######################################
# Threshold for Pixel Classification #
######################################
# Based on the plots above, it looks like only the Red parameter is a good one
# for single paramter thresholding.  HSV parameters better separate the data,
# but it looks like they need an interation term to separate on.
test.loc[test['Red'] > 100, 'thresh_red_pred'] = 'Aleurone'
test.loc[test['Red'] < 100, 'thresh_red_pred'] = 'Pericarp'
test.loc[test['Red'] == 0, 'thresh_red_pred'] = 'Background'

print(test)

print(sum(test['Label'] == test['thresh_red_pred']))

# According to these results, thresholding by red value is the best way to proceed
# I am skeptical though that this may be due to the way the data was collected,
# And aI also doubt this will hold true for yellow corn.

valid_copy = valid_image.copy()
valid_copy[valid_image[:,:,0] > 110] = [255,255,255]

plt.subplot(1,2,1)
plt.imshow(valid_image)
plt.title('Image without\nBackground')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(valid_copy)
plt.title('Image Aleurone\nRed Threshold')
plt.axis('off')
plt.tight_layout()
plt.show()

# Based on the above plot, it looks like red thresholding works really well,
# at least for this image. I have doubts about how generalizable this can be,
# but I think it shows that red can be an informative channel for white corn
# ML algorithms.
