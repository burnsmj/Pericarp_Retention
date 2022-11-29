#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:59:16 2022

@author: michael
"""
### Testing/Improving Image Masking
### Michael Burns
### 1/19/22

###################
# Import Packages #
###################
import numpy as np
from skimage import color
from skimage import filters
from skimage import morphology
from skimage import util
from skimage import io
import matplotlib.pyplot as plt
import pericarp_image_analysis_functions as piaf

def image_masking(image,
                  method = 'SV',
                  closable_area = 1500):
    """
    Parameters
    ----------
    image : NumPy array of a TIFF photo
        Should be an RGB photo, with a scale of 0-255.
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

    # Check Image Information #
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    # Convert Image to HSV Space #
    image_hsv = color.rgb2hsv(image[:,:,0:3])

    if method == 'SV':
        # Create SV Data for Image #
        sv_ratio = np.nan_to_num(image_hsv[:,:,1] / image_hsv[:,:,2]) * 100
        sv_mask = np.copy(image[:,:,0])

        # Threshold Images (Create Masks) #
        sv_mask[sv_ratio < piaf.image_auto_threshold(image_hsv)] = [0]
        sv_mask[sv_ratio >= piaf.image_auto_threshold(image_hsv)] = [1]

        # Close Holes in Mask #
        mask_no_sm_holes = morphology.binary_closing(sv_mask.astype(bool))
        mask_no_sm_objects = morphology.remove_small_objects(mask_no_sm_holes,
                                                             min_size = closable_area)

        # Alternative Way to Get Rid of Holes (Slower) #
        #mask_closed = morphology.area_closing(sv_mask,
        #                                      area_threshold = closable_area)
        #mask_opened = morphology.area_opening(mask_closed,
        #                                      area_threshold = closable_area)

    elif method == 'S':
        # Create S Mask Dataset #
        s_mask = np.copy(image[:,:,0])

        # Threshold Images (Create Masks) #
        s_mask[image_hsv[:,:,1] < piaf.image_auto_threshold(image_hsv)] = [0]
        s_mask[image_hsv[:,:,1] >= piaf.image_auto_threshold(image_hsv)] = [1]

        # Close Holes in Mask #
        mask_no_sm_holes = morphology.remove_small_holes(s_mask,
                                                    area_threshold = closable_area)
        mask_no_sm_objects = morphology.remove_small_objects(mask_no_sm_holes,
                                                             min_size = closable_area)

        # Alternative Way to Get Rid of Holes (Slower) #
        #mask_closed = morphology.area_closing(s_mask,
        #                                      area_threshold = closable_area)
        #mask_opened = morphology.area_opening(mask_closed,
        #                                      area_threshold = closable_area)
    else:
        print('Error: Please choose a method from the following list: SV, S')

    # Check Return Image Dimensions #
    assert(image.shape[0:2] == mask_no_sm_objects.shape), \
        'Masked image is not the same size as the original image'

    # Return Image Mask Data #
    return mask_no_sm_objects

#################
# Read in Image #
#################
image = io.imread('../junk.tif', plugin = 'pil')

plt.subplot(1,2,1)
plt.imshow(image)
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image_masking(image))
plt.axis('off')
plt.tight_layout()
plt.show()


# Option 1: change the closable area to 100 - likely to get a lot of non-kernel objects