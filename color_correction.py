#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:50:02 2022

@author: michael
"""
### Color Correction
### Michael Burns
### 1/13/22

# Purpose: To analyze a standardized color grid, determine how far off the
#           the colors are, create a transformation function, and apply it to
#           a new image.

###################
# Import Packages #
###################
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import glob

###############################
# Read in Color Checker Image #
###############################
color_checker_image = io.imread('../Data/Color_Correction/Color_Check_20220107.tif', plugin = 'pil')[:,:,0:3]

def grid_color_extraction(standard_image):
    """
    Parameters
    ----------
    standard_image : NumPy array of a TIFF photo
        Should be an RGB photo, with a scale of 0-255.
        The color grid used in the creation of this function was a 6x4 SpyderCheckr24
        

    Returns
    -------
    tuple of red, green, and blue mean values for an extracted square in a 
        standardized color grid

    Use
    ---
    red_mean, green_mean, blue_mean = grid_color_extraction(standard_image)

    Required Packages
    -----------------
    numpy as np
    """
    
    ###############################################
    # Create Coordinates at Center of Each Square #
    ###############################################
    first_coord = [600, 500]
    second_coord = [900, 800]
    
    ###########################################################
    # Initialize Lists for Mean Channel Values of Each Square #
    ###########################################################
    red_means = []
    green_means = []
    blue_means = []
    
    ################################################
    # Collecte Mean Channel Values for Each Square #
    ################################################
    for column in range(4):
        first_coord[1] = 500
        second_coord[1] = 800
        for row in range(6):
            red_means.append(round(np.mean(standard_image[first_coord[1]:second_coord[1],
                                                         first_coord[0]:second_coord[0], 0])))
            green_means.append(round(np.mean(standard_image[first_coord[1]:second_coord[1],
                                                           first_coord[0]:second_coord[0], 1])))
            blue_means.append(round(np.mean(standard_image[first_coord[1]:second_coord[1],
                                                          first_coord[0]:second_coord[0], 2])))
    
            first_coord[1] += 1450
            second_coord[1] += 1450
        first_coord[0] += 1450
        second_coord[0] += 1450
    
    ########################
    # Return Channel Means #
    ########################
    return (red_means, green_means, blue_means)

def calc_color_modifier(standard_image, show_shift = False, round_modifiers = True):
    """
    Parameters
    ----------
    standard_image : NumPy array of a TIFF photo
        Should be an RGB photo, with a scale of 0-255.
        The color grid used in the creation of this function was a 6x4 SpyderCheckr24
    show_shift : False by default
        Plots the shift in RGB values given the calculated modifier
    round_modifiers : True by default
        Rounds the returned modifiers so they can more easily be used in image
        correction

    Returns
    -------
    tuple of red, green, and blue modifiers based on means calculated in the
        grid color extraction function

    Use
    ---
    red_mod, green_mod, blue_mod = calc_color_modifier(standard_image)
    red_mod, green_mod, blue_mod = calc_color_modifier(standard_image, round_modifiers = False) # Will give a more quantitative value for modifiers

    Required Packages
    -----------------
    numpy as np
    pandas as pd
    matplotlib.pyplot as plt
    """
    #####################################
    # Read in Color Checker True Values #
    #####################################
    checker_key = pd.read_csv('../Data/Images/Color_Correction/Spyder_Checkr_24_Color_Key.csv', delimiter = ',', skiprows = 1)
    checker_key_24 = checker_key[checker_key.Spyder_24 == 'Y'].iloc[:, 6:9]
    
    #######################################
    # Extract Mean Values From Grid Image #
    #######################################
    mean_values = grid_color_extraction(standard_image)
    
    ##############################
    # Add Means to Key Dataframe #
    ##############################
    checker_key_24['R_bar'] = mean_values[0]
    checker_key_24['G_bar'] = mean_values[1]
    checker_key_24['B_bar'] = mean_values[2]
    
    ###################################################################
    # Determine Difference Between Actual and Observed Channel Values #
    ###################################################################
    checker_key_24['R_delta'] = checker_key_24.R - checker_key_24.R_bar
    checker_key_24['G_delta'] = checker_key_24.G - checker_key_24.G_bar
    checker_key_24['B_delta'] = checker_key_24.B - checker_key_24.B_bar
    
    #########################################
    # Calculate a Modifier for Each Channel #
    #########################################
    if round_modifiers is True:
        R_modifier = round(np.mean(checker_key_24.R_delta))
        G_modifier = round(np.mean(checker_key_24.G_delta))
        B_modifier = round(np.mean(checker_key_24.B_delta))
    else:
        R_modifier = np.mean(checker_key_24.R_delta)
        G_modifier = np.mean(checker_key_24.G_delta)
        B_modifier = np.mean(checker_key_24.B_delta)
    
    if show_shift is True:
        plt.subplot(2,3,1)
        plt.scatter(checker_key_24.R, checker_key_24.R_bar)
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Raw Red')
        plt.subplot(2,3,2)
        plt.scatter(checker_key_24.G, checker_key_24.G_bar)
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Raw Green')
        plt.subplot(2,3,3)
        plt.scatter(checker_key_24.B, checker_key_24.B_bar)
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Raw Blue')
        plt.subplot(2,3,4)
        plt.scatter(checker_key_24.R, checker_key_24.R_bar + round(R_modifier))
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Corrected Red')
        plt.subplot(2,3,5)
        plt.scatter(checker_key_24.G, checker_key_24.G_bar + round(G_modifier))
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Corrected Green')
        plt.subplot(2,3,6)
        plt.scatter(checker_key_24.B, checker_key_24.B_bar + round(B_modifier))
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Corrected Blue')
        plt.tight_layout()
        plt.show()
    
    ############################
    # Return Channel Modifiers #
    ############################
    return (R_modifier, G_modifier, B_modifier)

def image_color_correction(standard_image, image):
    """
    Parameters
    ----------
    standard_image : NumPy array of a TIFF photo
        Should be an RGB photo, with a scale of 0-255.
        The color grid used in the creation of this function was a 6x4 SpyderCheckr24
    image : NumPy array of a TIFF photo to be corrected
        Should be an RGB photo, with a scale of 0-255.
    Returns
    -------
    NumPy array of the modified image to be used in downstream analysis

    Use
    ---
    modified_image = image_color_correction(standard_image, image)

    Required Packages
    -----------------
    numpy as np
    pandas as pd
    """
    #############################
    # Calculate Image Modifiers #
    #############################
    img_modifiers = calc_color_modifier(standard_image)
    
    ###########################
    # Correct the Input Image #
    ###########################
    modified_image = np.copy(image)
    modified_image[:,:,0] += img_modifiers[0]
    modified_image[:,:,1] += img_modifiers[1]
    modified_image[:,:,2] += img_modifiers[2]
    
    #############################
    # Return the Modified Image #
    #############################
    return modified_image

#################
# Sanity Checks #
#################
calc_color_modifier(color_checker_image, True)

plt.subplot(1,2,1)
plt.imshow(color_checker_image)
plt.axis('off')
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(image_color_correction(color_checker_image, color_checker_image))
plt.axis('off')
plt.title('Modified')
plt.show()

########################
# Within Day Variation #
########################
images = sorted(glob.glob('../Data/Color_Correction/Color_Check_2021*_*.tif')) # List images that are all taken within a day
print(images) # Write out the list of file names

times = []
red_modifiers = []
green_modifiers = []
blue_modifiers = []

for image in images:
    picture = io.imread(image, plugin = 'pil')[:,:,0:3] # Read in image
    picture_copy = picture.copy()
    time = image[-8:-4] # Extract the time
    if time.startswith('_'):
        time = time[-3:] # Correct the time if it only contains 3 digits

    times.append(time) # Save time of day

    # Sanity Check for Position Start
    first_coord = [600, 500]
    second_coord = [900, 800]

    for column in range(4):
        first_coord[1] = 500
        second_coord[1] = 800
        for row in range(6):
            picture_copy[first_coord[1]:second_coord[1],
                    first_coord[0]:second_coord[0], 0] = 0
            picture_copy[first_coord[1]:second_coord[1],
                    first_coord[0]:second_coord[0], 1] = 0
            picture_copy[first_coord[1]:second_coord[1],
                    first_coord[0]:second_coord[0], 2] = 0

            first_coord[1] += 1450
            second_coord[1] += 1450
        first_coord[0] += 1450
        second_coord[0] += 1450

    plt.imshow(picture_copy)
    plt.show()
    # Sanity Check for Position End

    red_modifiers.append(calc_color_modifier(picture, True, False)[0])
    green_modifiers.append(calc_color_modifier(picture, False, False)[1])
    blue_modifiers.append(calc_color_modifier(picture, False, False)[2])

print('Within Day Variance:')
print('Red Mod: ' + str(np.mean(red_modifiers)) + ' +/- ' + str(np.var(red_modifiers)))
print('Green Mod: ' + str(np.mean(green_modifiers)) + ' +/- ' + str(np.var(green_modifiers)))
print('Blue Mod: ' + str(np.mean(blue_modifiers)) + ' +/- ' + str(np.var(blue_modifiers)))

########################
# Across Day Variation #
########################
images = sorted(glob.glob('../Data/Color_Correction/Color_Check_' + ('[0-9]' * 8) + '.tif')) # List images that are all taken within a day
print(images) # Write out the list of file names

times = []
red_modifiers = []
green_modifiers = []
blue_modifiers = []

for image in images:
    picture = io.imread(image, plugin = 'pil')[:,:,0:3] # Read in image
    picture_copy = picture.copy()
    time = image[-8:-4] # Extract the time
    if time.startswith('_'):
        time = time[-3:] # Correct the time if it only contains 3 digits

    times.append(time) # Save time of day

    # Sanity Check for Position Start
    first_coord = [600, 500]
    second_coord = [900, 800]

    for column in range(4):
        first_coord[1] = 500
        second_coord[1] = 800
        for row in range(6):
            picture_copy[first_coord[1]:second_coord[1],
                    first_coord[0]:second_coord[0], 0] = 0
            picture_copy[first_coord[1]:second_coord[1],
                    first_coord[0]:second_coord[0], 1] = 0
            picture_copy[first_coord[1]:second_coord[1],
                    first_coord[0]:second_coord[0], 2] = 0

            first_coord[1] += 1450
            second_coord[1] += 1450
        first_coord[0] += 1450
        second_coord[0] += 1450

    plt.imshow(picture_copy)
    plt.show()
    # Sanity Check for Position End

    red_modifiers.append(calc_color_modifier(picture, True, False)[0])
    green_modifiers.append(calc_color_modifier(picture, False, False)[1])
    blue_modifiers.append(calc_color_modifier(picture, False, False)[2])

print('Across Day Variance:')
print('Red Mod: ' + str(np.mean(red_modifiers)) + ' +/- ' + str(np.var(red_modifiers)))
print('Green Mod: ' + str(np.mean(green_modifiers)) + ' +/- ' + str(np.var(green_modifiers)))
print('Blue Mod: ' + str(np.mean(blue_modifiers)) + ' +/- ' + str(np.var(blue_modifiers)))

# NEXT: Need to find a way to pick a color correction image based on date associated with a kernel image
