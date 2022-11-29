#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:16:33 2021

@author: michael
"""
### Image Kernel Detection
### Michael Burns
### 10/11/21

# Purpose: To create functions that will detect the location of kernels in
#          order to more easily remove the background from the images

# NOTE: Should add assertions to the functions

###################
# Import Packages #
###################
import glob
import numpy as np
import pandas as pd
from skimage import io
from skimage import color
from skimage import filters
from skimage import util
from skimage import measure
from skimage import morphology
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmin

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

#plt.subplot(1,2,1)
#plt.imshow(image_rgb)
#plt.axis('off')
#plt.subplot(1,2,2)
#plt.imshow(image_blur)
#plt.axis('off')
#plt.tight_layout()
#plt.show()

def image_auto_threshold(hsv_image, 
                         method = 'SV',
                         stringency = 0):
    
    ######################
    # Determine SV Ratio #
    ######################
    image_sv = np.nan_to_num(hsv_image[:,:,1] / hsv_image[:,:,2]) * 100
    
    ##################################
    # Determine Density of SV Ratios #
    ##################################
    density_info = np.histogram(image_sv.ravel(), density = True, bins = 256)
    
    ########################################
    # Find and Skip Past the Peak SV Ratio #
    ########################################
    max_value = max(density_info[0])
    max_index = np.where(density_info[0] == max_value)[0][0]
    
    #######################################
    # Determine the Appropriate Threshold #
    #######################################
    ideal_threshold = density_info[1][argrelmin(density_info[0])[0][argrelmin(density_info[0])[0] > max_index][stringency]]
    
    ##########################
    # Return Threshold Value #
    ##########################
    return ideal_threshold

def image_masking(image,
                  method = 'SV',
                  closable_area = 1500):
    """
    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGB photo, with a scale of 0-1 rather than 0-255.
    method : string | options: 'SV', 'S'
        A string indicating the preferred masking metric. If equal to SV, the
        function will mask based on (saturation / value) * 100 ratio, and if
        set to S, the image will be masked by saturation.
    threshold : Integer
        Pixels with a value (determined by method) greater than threshold
        will be considered foreground and turned white (255,255,255), whereas
        pixels with a value less than threshold will be considered background
        and turned black (0,0,0). This value may need to be changed for
        different pictures, on different days, or for different colored corn.
        The default is 12.
    closable_area : Integer
        The number of clustered pixels that can be misidentified in a given
        location and turned into the surrounding classification.
        The default is 200.

    Returns
    -------
    sv_mask_closed : A binary image with dimensions NxM to match the input
    images NxMx3 dimensions.  White is foregrounnd, black is background.
    --OR--
    s_mask_closed : A binary image with dimensions NxM to match the input
    images NxMx3 dimensions.  White is foregrounnd, black is background.

    Use
    ---
    image_masking(<image_name>) : Returns a binary image described above.
    image_masking(<image_name>, threshold = 32) : Returns a binary image
        described above with more values considered background.

    Required Packages
    -----------------
    numpy as np
    color from skimage
    filters from skimage
    morphology from skimage
    util from skimage
    """

    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    ##############################
    # Convert Image to HSV Space #
    ##############################
    image_hsv = color.rgb2hsv(image[:,:,0:3])

    if method == 'SV':
        ############################
        # Create SV Data for Image #
        ############################
        sv_ratio = np.nan_to_num(image_hsv[:,:,1] / image_hsv[:,:,2]) * 100
        sv_mask = np.copy(image[:,:,0])

        ###################################
        # Threshold Images (Create Masks) #
        ###################################
        sv_mask[sv_ratio < image_auto_threshold(image_hsv)] = [0]
        sv_mask[sv_ratio >= image_auto_threshold(image_hsv)] = [1]

        #######################
        # Close Holes in Mask #
        #######################
        mask_no_sm_holes = morphology.remove_small_holes(sv_mask.astype(bool),
                                                    area_threshold = closable_area)
        mask_no_sm_objects = morphology.remove_small_objects(mask_no_sm_holes,
                                                             min_size = closable_area)

        ################################################
        # Alternative Way to Get Rid of Holes (Slower) #
        ################################################
        #mask_closed = morphology.area_closing(sv_mask,
        #                                      area_threshold = closable_area)
        #mask_opened = morphology.area_opening(mask_closed,
        #                                      area_threshold = closable_area)
        
    elif method == 'S':
        #########################
        # Create S Mask Dataset #
        #########################
        s_mask = np.copy(image[:,:,0])

        ###################################
        # Threshold Images (Create Masks) #
        ###################################
        s_mask[image_hsv[:,:,1] < image_auto_threshold(image_hsv)] = [0]
        s_mask[image_hsv[:,:,1] >= image_auto_threshold(image_hsv)] = [1]
        
        #######################
        # Close Holes in Mask #
        #######################
        mask_no_sm_holes = morphology.remove_small_holes(s_mask,
                                                    area_threshold = closable_area)
        mask_no_sm_objects = morphology.remove_small_objects(mask_no_sm_holes,
                                                             min_size = closable_area)

        ################################################
        # Alternative Way to Get Rid of Holes (Slower) #
        ################################################
        #mask_closed = morphology.area_closing(s_mask,
        #                                      area_threshold = closable_area)
        #mask_opened = morphology.area_opening(mask_closed,
        #                                      area_threshold = closable_area)
    else:
        print('Error: Please choose a method from the following list: SV, S')

    #################################
    # Check Return Image Dimensions #
    #################################
    assert(image.shape[0:2] == mask_no_sm_objects.shape), \
        'Masked image is not the same size as the original image'

    ##########################
    # Return Image Mask Data #
    ##########################
    return mask_no_sm_objects

def image_background_removal(image,
                             method = 'SV',
                             closable_area = 1500):
    """
    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGB photo, with a scale of 0-1 rather than 0-255.
    method : string | options: 'SV', 'S'
        A string indicating the preferred masking metric. If equal to SV, the
        function will mask based on (saturation / value) * 100 ratio, and if
        set to S, the image will be masked by saturation.
    threshold : Integer
        Pixels with a value (determined by method) greater than threshold
        will be considered foreground and turned white (255,255,255), whereas
        pixels with a value less than threshold will be considered background
        and turned black (0,0,0). This value may need to be changed for
        different pictures, on different days, or for different colored corn.
        The default is 12.
    closable_area : Integer
        The number of clustered pixels that can be misidentified in a given
        location and turned into the surrounding classification.
        The default is 200.

    Returns
    -------
    image : An image with dimensions NxMx3 to match the input
        images NxMx3 dimensions.  foreground has retained its color, background
        has been turned black.

    Use
    ---
    labelled_image(<image_name>) : Returns an image described above
    labelled_image(<image_name>, threshold = 32) : Returns an image
        described above with more values considered background.

    Required Packages
    -----------------
    numpy as np
    color from skimage
    filters from skimage
    morphology from skimage
    util from skimage
    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    ##################################
    # Mask Image According to Method #
    ##################################
    mask = image_masking(image,
                         method = method,
                         closable_area = closable_area)

    ####################################
    # Create Images without Background #
    ####################################
    no_background = image.copy()
    no_background[mask == 0] = [0,0,0]

    #################################
    # Check Return Image Dimensions #
    #################################
    assert(image.shape == no_background.shape), \
        'Output image is not the same size as the original image'

    ###################################
    # Return Image Without Background #
    ###################################
    return no_background

#####################
# Label Mask Images #
#####################
def image_labelling(image,
                    method = 'SV',
                    closable_area = 1500):
    """
    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGB photo, with a scale of 0-1 rather than 0-255.
    method : string | options: 'SV', 'S'
        A string indicating the preferred masking metric. If equal to SV, the
        function will mask based on (saturation / value) * 100 ratio, and if
        set to S, the image will be masked by saturation.
    threshold : Integer
        Pixels with a value (determined by method) greater than threshold
        will be considered foreground and turned white (255,255,255), whereas
        pixels with a value less than threshold will be considered background
        and turned black (0,0,0). This value may need to be changed for
        different pictures, on different days, or for different colored corn.
        The default is 12.
    closable_area : Integer
        The number of clustered pixels that can be misidentified in a given
        location and turned into the surrounding classification.
        The default is 200.

    Returns
    -------
    labelled_image : an NxM image (where the input image is NxMx3), of integer
        values.  The integers are classifications of clustered pixels.
        0 = background, 1 = kernel 1, 2 = kernel 2, ...

    Use
    ---
    labelled_image(<image_name>) : Returns a 2D image described above
    labelled_image(<image_name>, threshold = 32) : Returns a 2D image
        described above with more values considered background.

    Required Packages
    -----------------
    numpy as np
    color from skimage
    filters from skimage
    morphology from skimage
    util from skimage
    measure from skimage
    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    #######################
    # Create Binary Image #
    #######################
    mask_image = image_masking(image,
                               method = method,
                               closable_area = closable_area)

    #########################
    # Create Labelled Image #
    #########################
    labelled_image = measure.label(mask_image)

    #################################
    # Check Return Image Dimensions #
    #################################
    assert(image.shape[0:2] == labelled_image.shape), \
        'Output image is not the same size as the original image'

    #########################
    # Return Labelled Image #
    #########################
    return labelled_image

##########################################
# Extract a given labelled kernel's data #
##########################################
def indiv_kernel_data_extraction(image, kernel_number = 1):
    """
    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGB photo, with a scale of 0-1 rather than 0-255.
    kernel_number : An integer of the labelled kernel of interest
        Should be between 1 and the total number of kernels.
        Label 0 is the background

    Returns
    -------
    kernel_data : an 8xN pandas dataframe (where N is the number of pixels)
        containing the indecies of the pixel (in X and Y), the RGB, and HSV
        values (each channel has its own channel).

    Use
    ---
    indiv_kernel_extraction(<image_name>) : Returns a dataframe with RGB and HSV data
        as well as pixel coordinates for one kernel.
    for kernel in range(1,11):
        indiv_kernel_extraction(<image_name>, kernel_number = kernel) : Returns
            the data (Indecies, RGB, HSV) for kernels 1 through 10 in given image.

    Required Packages
    -----------------
    numpy as np
    color from skimage
    filters from skimage
    morphology from skimage
    util from skimage
    measure from skimage
    pandas as pd
    """
    #######################################
    # Create Labelled Image and HSV Image #
    #######################################
    labelled_image = image_labelling(image)
    image_hsv = color.rgb2hsv(image)
    image_lab = color.rgb2lab(image / 255) # rgb2lab expects data to be between 0 and 1

    #################################################
    # Create Flat Numpy Arrays of Indicies and Data #
    #################################################
    np_ind = np.fliplr(np.array(np.where(labelled_image == kernel_number)).transpose())
    np_rgb = image[labelled_image == kernel_number]
    np_hsv = image_hsv[labelled_image == kernel_number] # Make sure this is consistent with training set
    np_lab = image_lab[labelled_image == kernel_number]

    ###################################
    # Stack Flat Numpy Arrays of Data #
    ###################################
    np_kernel = np.hstack((np_ind, np_rgb, np_hsv, np_lab))

    #######################################
    # Compiling Data in Pandas Data Frame #
    #######################################
    kernel_data = pd.DataFrame(np_kernel, columns = ['X', 'Y',
                                                     'Red',
                                                     'Green',
                                                     'Blue',
                                                     'Hue',
                                                     'Saturation',
                                                     'Value',
                                                     'Luminosity',
                                                     'A_Axis',
                                                     'B_Axis'], index = None)

    #########################
    # Return Tabulated Data #
    #########################
    return kernel_data

###################################
# Write out Images of each Kernel #
###################################
def indiv_kernel_image_extraction(image,
                                  padding = 10,
                                  fname = 'test_file',
                                  ftype = 'tif'):
    labelled_image = image_labelling(image)
    kernel_count = np.unique(labelled_image)
    
    for kernel in kernel_count[1:]:
        coords = np.where(labelled_image == kernel)
        
        kernel_image = image_background_removal(image)[min(coords[0]) - padding : max(coords[0]) + padding,
                                                     min(coords[1]) - padding : max(coords[1]) + padding,:]
        
        io.imsave(str(fname) + '_' + str(kernel) + '.' + str(ftype), kernel_image)
        
#indiv_kernel_image_extraction(image_rgb)

###############
# Show Images #
###############
SHOW_PLOTS = False

if SHOW_PLOTS is True:
    print('Starting Plotting')
    plt.subplot(1,5,1)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Original\nImage')
    plt.subplot(1,5,2)
    plt.imshow(image_blur)
    plt.axis('off')
    plt.title('Blurred\nImage')
    plt.subplot(1,5,3)
    plt.imshow(image_masking(image_blur), cmap = 'gray')
    plt.axis('off')
    plt.title('Binary\nImage')
    plt.subplot(1,5,4)
    plt.imshow(image_labelling(image_blur))
    plt.axis('off')
    plt.title('Labelled\nImage')
    plt.subplot(1,5,5)
    plt.imshow(image_background_removal(image_blur))
    plt.axis('off')
    plt.title('Image Without\nBackground')
    plt.tight_layout()
    plt.show()

###############
# Show Labels #
###############
    labelled_image_var = image_labelling(image_blur)
    for i in range(1, 11):
        image_copy = image_blur.copy()
        image_copy[labelled_image_var != i] = [0,0,0]
        image_copy[labelled_image_var == i] = [255,255,255]
        plt.subplot(2,5,i)
        plt.imshow(image_copy)
        plt.axis('off')
        plt.title('Label: ' + str(i))
    plt.tight_layout()
    plt.show()

######################################################
# Trial of Pipeline on Images of Different Genotypes #
######################################################
TRIAL = False

if TRIAL is True:
    images = sorted(glob.glob('../Data/Images/Trials/Multi_Genotype_Images/*.tif'))
    print(images)
    
    for image in images:
        input_image = io.imread(image, plugin = 'pil')
        
        input_blur = util.img_as_ubyte(filters.gaussian(input_image,
                                       sigma = 5,
                                       multichannel = True,
                                       truncate = 2))
        
        plt.subplot(1,5,1)
        plt.imshow(input_image)
        plt.axis('off')
        plt.title('Original\nImage')
        plt.subplot(1,5,2)
        plt.imshow(input_blur)
        plt.axis('off')
        plt.title('Blurred\nImage')
        plt.subplot(1,5,3)
        plt.imshow(image_masking(input_blur), cmap = 'gray')
        plt.axis('off')
        plt.title('Binary Image\n Threshold: ' + str(round(image_auto_threshold(color.rgb2hsv(input_blur[:,:,0:3])), 2)))
        plt.subplot(1,5,4)
        plt.imshow(image_labelling(input_blur))
        plt.axis('off')
        plt.title('Labels: ' + str(image_labelling(input_blur).max()) + '\n')
        plt.subplot(1,5,5)
        plt.imshow(image_background_removal(input_blur))
        plt.axis('off')
        plt.title('Image Without\nBackground')
        plt.tight_layout()
        plt.show()
