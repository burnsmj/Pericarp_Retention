#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:47:11 2021

@author: michael
"""
### Image Background Detection
### Michael Burns
### 10/7/21

"""

Purpose: To learn how to segment an image
         To better learn scikit image
         To remove the background of the kernel images to reduce the amount of noise

"""

###################
# Import Packages #
###################
from skimage import io
from skimage import color
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl



######################## 
# Plotting Adjustments #
########################
mpl.rcParams['figure.dpi'] = 300 # Changing default figure resolution (increase)



#################
# Read in Image #
#################
image_rgb = io.imread('../Data/Images/Trials/Poststain_White_MB_H2O.png')

image_rgb = filters.gaussian(image_rgb, sigma = 5, multichannel = True, truncate = 2)



##############################
# Convert Image to Lab Space #
##############################
image_lab = color.rgb2lab(image_rgb[:,:,0:3])



##############################
# Convert Image to HSV Space #
##############################
image_hsv = color.rgb2hsv(image_rgb[:,:,0:3])



##########################
# Combine Raw Image Data #
##########################
avg_intensity = np.mean(image_rgb[:,:,0:3], axis = 2)
var_intensity = np.var(image_rgb[:,:,0:3], axis = 2, ddof = 1)



##########################
# Combine Lab Image Data #
##########################
ab_avg = np.mean(image_lab[:,:,1:3], axis = 2)
ab_var = np.var(image_lab[:,:,1:3], axis = 2, ddof = 1)



##########################
# Analyze HSV Image Data #
##########################
sv_ratio = image_hsv[:,:,1] / image_hsv[:,:,2]



###############
# Show Images #
###############
plt.subplot(3,5,1)
plt.imshow(image_rgb[:,:,0], cmap = 'gray')
plt.title('R')
plt.axis('off')
plt.subplot(3,5,2)
plt.imshow(image_rgb[:,:,1], cmap = 'gray')
plt.title('G')
plt.axis('off')
plt.subplot(3,5,3)
plt.imshow(image_rgb[:,:,2], cmap = 'gray')
plt.title('B')
plt.axis('off')
plt.subplot(3,5,4)
plt.imshow(avg_intensity, cmap = 'gray')
plt.title('Avg Int')
plt.axis('off')
plt.subplot(3,5,5)
plt.imshow(var_intensity, cmap = 'gray')
plt.title('Var Int')
plt.axis('off')
plt.subplot(3,5,6)
plt.imshow(image_lab[:,:,0], cmap = 'gray')
plt.title('L')
plt.axis('off')
plt.subplot(3,5,7)
plt.imshow(image_lab[:,:,1], cmap = 'gray')
plt.title('A')
plt.axis('off')
plt.subplot(3,5,8)
plt.imshow(image_lab[:,:,2], cmap = 'gray')
plt.title('B')
plt.axis('off')
plt.subplot(3,5,9)
plt.imshow(ab_avg, cmap = 'gray')
plt.title('AB Avg')
plt.axis('off')
plt.subplot(3,5,10)
plt.imshow(ab_var, cmap = 'gray')
plt.title('AB Var')
plt.axis('off')
plt.subplot(3,5,11)
plt.imshow(image_hsv[:,:,0], cmap = 'gray')
plt.title('H')
plt.axis('off')
plt.subplot(3,5,12)
plt.imshow(image_hsv[:,:,1], cmap = 'gray')
plt.title('S')
plt.axis('off')
plt.subplot(3,5,13)
plt.imshow(image_hsv[:,:,2], cmap = 'gray')
plt.title('V')
plt.axis('off')
plt.subplot(3,5,14)
plt.imshow(sv_ratio, cmap = 'gray')
plt.title('S/V')
plt.axis('off')
plt.tight_layout()
plt.show()



######################
# Plot Distributions #
######################
plt.subplot(3,5,1)
plt.hist(image_rgb[:,:,0].ravel(), bins = 255)
plt.title('R')
plt.axis('off')
plt.subplot(3,5,2)
plt.hist(image_rgb[:,:,1].ravel(), bins = 255)
plt.title('G')
plt.axis('off')
plt.subplot(3,5,3)
plt.hist(image_rgb[:,:,2].ravel(), bins = 255)
plt.title('B')
plt.axis('off')
plt.subplot(3,5,4)
plt.hist(avg_intensity.ravel(), bins = 255)
plt.title('Avg Int')
plt.axis('off')
plt.subplot(3,5,5)
plt.hist(var_intensity.ravel(), bins = 255)
plt.title('Var Int')
plt.axis('off')
plt.subplot(3,5,6)
plt.hist(image_lab[:,:,0].ravel(), bins = 255)
plt.title('L')
plt.axis('off')
plt.subplot(3,5,7)
plt.hist(image_lab[:,:,1].ravel(), bins = 255)
plt.title('A')
plt.axis('off')
plt.subplot(3,5,8)
plt.hist(image_lab[:,:,2].ravel(), bins = 255)
plt.title('B')
plt.axis('off')
plt.subplot(3,5,9)
plt.hist(ab_avg.ravel(), bins = 255)
plt.title('AB Avg')
plt.axis('off')
plt.subplot(3,5,10)
plt.hist(ab_var.ravel(), bins = 255)
plt.axis('off')
plt.title('AB Var')
plt.subplot(3,5,11)
plt.hist(image_hsv[:,:,0].ravel(), bins = 255)
plt.title('H')
plt.axis('off')
plt.subplot(3,5,12)
plt.hist(image_hsv[:,:,1].ravel(), bins = 255)
plt.title('S')
plt.axis('off')
plt.subplot(3,5,13)
plt.hist(image_hsv[:,:,2].ravel(), bins = 255)
plt.title('V')
plt.axis('off')
plt.subplot(3,5,14)
plt.hist(sv_ratio.ravel(), bins = 255)
plt.title('S / V Ratio')
plt.axis('off')
plt.tight_layout()
plt.show()


############################
# Plot S, V, and S/V Ratio #
############################
plt.subplot(2,3,1)
plt.hist(image_rgb[:,:,0].ravel(), bins = 256)
plt.title('Red')
plt.subplot(2,3,2)
plt.hist(image_rgb[:,:,1].ravel(), bins = 256)
plt.title('Green')
plt.subplot(2,3,3)
plt.hist(image_rgb[:,:,2].ravel(), bins = 256)
plt.title('Blue')
plt.subplot(2,3,4)
plt.hist(image_hsv[:,:,1].ravel(), bins = 256)
plt.title('Saturation')
plt.ylim(0, 100000)
plt.subplot(2,3,5)
plt.hist(image_hsv[:,:,2].ravel(), bins = 256)
plt.title('Value')
plt.ylim(0, 100000)
plt.subplot(2,3,6)
plt.hist(sv_ratio.ravel(), bins = 256)
plt.title('S / V Ratio')
plt.ylim(0, 100000)
plt.tight_layout()
plt.show()


############################
# Plotting Processed Image #
############################
#image_rgb[avg_intensity > 180] = [0,0,0,255]
#image_rgb[var_intensity < 15] = [0,0,0,255]
image_rgb[sv_ratio < 0.15] = [0,0,0,255]

plt.imshow(image_rgb)
plt.title('No Background?')
plt.axis('off')
plt.show()

