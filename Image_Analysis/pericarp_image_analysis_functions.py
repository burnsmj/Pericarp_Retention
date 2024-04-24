#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:25:04 2022

@author: michael
"""
### Pericarp Image Analysis Functions
### Michael Burns
### 1/18/22

# Purpose: To aggregate all of the functions associated with pericarp image
#           analysis into one place.  With all of the functions in one script,
#           I can use other scripts for specific analyses, and only need to
#           import one script or a few functions from a given script.  This
#           will help the analysis scripts stay cleaner as they will not start
#           with a list of relevant functions.

# Goal: To keep a well documented (comments and doc strings) list of functions
#       that are relevant to the pericarp image analysis project.

######################
# Importing Packages #
######################
import time
from random import seed
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from skimage import color
from skimage import measure
from skimage import morphology
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn import naive_bayes # Base level model approach
from sklearn import svm # Complex, accurate, slow approach
from sklearn import neighbors # Clustering approach
from sklearn import tree # Decision tree approach
from sklearn import ensemble # Multimodel approach
from sklearn import linear_model # Neural network approach
from scipy.signal import argrelmin
#import xgboost as xgb
import matplotlib.pyplot as plt

##################################
# Functions for Color Correction #
##################################
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

    # Create Coordinates at Center of Each Square #
    first_coord = [600, 500]
    second_coord = [900, 800]

    # Initialize Lists for Mean Channel Values of Each Square #
    red_means = []
    green_means = []
    blue_means = []

    # Collecte Mean Channel Values for Each Square #
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

    # Return Channel Means #
    return (red_means, green_means, blue_means)

def calc_color_modifier(standard_image, image_key, show_shift = False, round_modifiers = True):
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
    red_mod, green_mod, blue_mod = calc_color_modifier(standard_image, round_modifiers = False)

    Required Packages
    -----------------
    numpy as np
    pandas as pd
    matplotlib.pyplot as plt
    """
    # Read in Color Checker True Values #
    checker_key = image_key
    checker_key_24 = checker_key[checker_key.Spyder_24 == 'Y'].iloc[:, 6:9]

    # Extract Mean Values From Grid Image #
    mean_values = grid_color_extraction(standard_image)

    # Add Means to Key Dataframe #
    checker_key_24['R_bar'] = mean_values[0]
    checker_key_24['G_bar'] = mean_values[1]
    checker_key_24['B_bar'] = mean_values[2]

    # Determine Difference Between Actual and Observed Channel Values #
    checker_key_24['R_delta'] = checker_key_24.R - checker_key_24.R_bar
    checker_key_24['G_delta'] = checker_key_24.G - checker_key_24.G_bar
    checker_key_24['B_delta'] = checker_key_24.B - checker_key_24.B_bar

    # Calculate a Modifier for Each Channel #
    if round_modifiers is True:
        r_modifier = round(np.mean(checker_key_24.R_delta))
        g_modifier = round(np.mean(checker_key_24.G_delta))
        b_modifier = round(np.mean(checker_key_24.B_delta))
    else:
        r_modifier = np.mean(checker_key_24.R_delta)
        g_modifier = np.mean(checker_key_24.G_delta)
        b_modifier = np.mean(checker_key_24.B_delta)

    # If asked for, return a plot of the shifted values #
    if show_shift is True:
        plt.subplot(1,3,1)
        plt.scatter(checker_key_24.R, checker_key_24.R_bar)
        plt.scatter(checker_key_24.R, checker_key_24.R_bar + round(r_modifier))
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Red')
        plt.subplot(1,3,2)
        plt.scatter(checker_key_24.G, checker_key_24.G_bar)
        plt.scatter(checker_key_24.G, checker_key_24.G_bar + round(g_modifier))
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Green')
        plt.subplot(1,3,3)
        plt.scatter(checker_key_24.B, checker_key_24.B_bar)
        plt.scatter(checker_key_24.B, checker_key_24.B_bar + round(b_modifier))
        plt.xlim(0,255)
        plt.ylim(0,255)
        plt.plot(np.linspace(0,255, 10), np.linspace(0,255, 10))
        plt.title('Blue')
        plt.tight_layout()
        plt.show()

    # Return Channel Modifiers #
    return (r_modifier, g_modifier, b_modifier)

def image_color_correction(standard_image, image_key, image):
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
    # Calculate Image Modifiers #
    img_modifiers = calc_color_modifier(standard_image, image_key)

    # Correct the Input Image #
    modified_image = np.copy(image)
    modified_image[:,:,0] += img_modifiers[0]
    modified_image[:,:,1] += img_modifiers[1]
    modified_image[:,:,2] += img_modifiers[2]
    
    modified_image[modified_image[:,:,0] < img_modifiers[0]] = 255
    modified_image[modified_image[:,:,1] < img_modifiers[1]] = 255
    modified_image[modified_image[:,:,2] < img_modifiers[2]] = 255

    # Return the Modified Image #
    return modified_image

###################################
# Functions for Kernel Extraction #
###################################
def image_auto_threshold(hsv_image,
                         stringency = 0):
    """
    Parameters
    ----------
    hsv_image : NumPy array of a TIFF photo that has been coverted to HSV
    stringency : default is zero.
        the stringency of the given threshold.  The position in a density list
        determined for each image.

    Returns
    -------
    ideal_threshold : a threshold to use when thresholding S/V values of an
        image to remove background.  This is based on actual image values, so
        it should be more generalizeable and less arbitrary.

    Use
    ---
    ideal_threshold = image_auto_threshold(hsv_image)

    Required Packages
    -----------------
    numpy as np
    from scipy.signal import argrelmin
    """

    # Determine SV Ratio #
    image_sv = np.nan_to_num(hsv_image[:,:,1] / hsv_image[:,:,2]) * 100

    # Determine Density of SV Ratios #
    density_info = np.histogram(image_sv.ravel(), density = True, bins = 256)

    # Find and Skip Past the Peak SV Ratio #
    max_value = max(density_info[0])
    max_index = np.where(density_info[0] == max_value)[0][0]

    # Determine the Appropriate Threshold #
    ideal_threshold = density_info[1][argrelmin(density_info[0])[0][argrelmin(density_info[0])[0] > max_index][stringency]]

    # Return Threshold Value #
    return ideal_threshold

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
        The default is 1500.

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
        # Change values of zero to a very small number
        image_hsv[image_hsv[:,:,2] == 0] = 0.00001
        # Create SV Data for Image #
        sv_ratio = np.nan_to_num(image_hsv[:,:,1] / image_hsv[:,:,2]) * 100
        sv_mask = np.copy(image[:,:,0])

        # Threshold Images (Create Masks) #
        sv_mask[sv_ratio < image_auto_threshold(image_hsv)] = [0]
        sv_mask[sv_ratio >= image_auto_threshold(image_hsv)] = [1]

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
        s_mask[image_hsv[:,:,1] < image_auto_threshold(image_hsv)] = [0]
        s_mask[image_hsv[:,:,1] >= image_auto_threshold(image_hsv)] = [1]

        # Close Holes in Mask #
        mask_no_sm_holes = morphology.binary_closing(s_mask.astype(bool))
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

def image_background_removal(image,
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
        The default is 1500.

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
    # Check Image Information #
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    # Mask Image According to Method #
    mask = image_masking(image,
                         method = method,
                         closable_area = closable_area)

    # Create Images without Background #
    no_background = image.copy()
    no_background[mask == 0] = [0,0,0]

    # Check Return Image Dimensions #
    assert(image.shape == no_background.shape), \
        'Output image is not the same size as the original image'

    # Return Image Without Background #
    return no_background

def image_labelling(image,
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
    # Check Image Information #
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    # Create Binary Image #
    mask_image = image_masking(image,
                               method = method,
                               closable_area = closable_area)

    # Create Labelled Image #
    labelled_image = measure.label(mask_image)

    # Check Return Image Dimensions #
    assert(image.shape[0:2] == labelled_image.shape), \
        'Output image is not the same size as the original image'

    # Return Labelled Image #
    return labelled_image

def indiv_kernel_data_extraction(image, kernel_number = 1):
    """
    Parameters
    ----------
    image : NumPy array of a TIFF photo
        Should be an RGB photo, with a scale of 0-255.
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
    # Create Labelled Image and HSV Image #
    labelled_image = image_labelling(image)
    image_hsv = color.rgb2hsv(image)
    image_lab = color.rgb2lab(image / 255) # rgb2lab expects data to be between 0 and 1

    # Create Flat Numpy Arrays of Indicies and Data #
    np_ind = np.fliplr(np.array(np.where(labelled_image == kernel_number)).transpose())
    np_rgb = image[labelled_image == kernel_number]
    np_hsv = image_hsv[labelled_image == kernel_number] # Confirm consistent with training set
    np_lab = image_lab[labelled_image == kernel_number]

    # Stack Flat Numpy Arrays of Data #
    np_kernel = np.hstack((np_ind, np_rgb, np_hsv, np_lab))

    # Compiling Data in Pandas Data Frame #
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

    # Return Tabulated Data #
    return kernel_data

def indiv_kernel_image_extraction(image,
                                  padding = 10,
                                  fpath = 'test_file',
                                  ftype = 'tif'):
    """
    Parameters
    ----------
    image : NumPy array of a TIFF photo
        Should be an RGB photo, with a scale of 0-255.
    padding : the number of pixels to leave on each of the cardinal directions
        of a kernel.
    fpath : a file path to save the pictures in
    ftype : a file suffix to tell the computer what type of picture it is.

    Returns
    -------
    Automatically saves each kernel image individually to the desired path with
        and increasing number associated with it based on how many kernels are
        in the image.

    Use
    ---
    indiv_kernel_image_extraction(image)

    Required Packages
    -----------------
    numpy as np
    from skimage import io
    """
    # Label each kernel and count the number of kernels #
    labelled_image = image_labelling(image)
    kernel_count = np.unique(labelled_image)

    # For each kernel, extract it and save it as a file #
    for kernel in tqdm(kernel_count[1:]):
        coords = np.where(labelled_image == kernel)

        kernel_image = image_background_removal(image)[min(coords[0]) - padding : max(coords[0]) + padding,
                                                     min(coords[1]) - padding : max(coords[1]) + padding,:]

        io.imsave(str(fpath) + '_' + str(kernel) + '.' + str(ftype), kernel_image)

######################################
# Functions for Pixel Classification #
######################################
def training_data_distributions(train_data,
                                channel_names = ('Red', 'Green', 'Blue',
                                                 'Hue', 'Saturation', 'Value',
                                                 'Luminosity', 'A_Axis', 'B_Axis'),
                                alpha = 0.7):
    """
    Parameters
    ----------
    train_data : a pandas dataframe to be used in cross validated machine
        learning. Should not contain test data.
    channel_names : a [list] of all possible columns to use
    alpha : how transparent to make the images

    Returns
    -------
    A plot of histograms of each of the features to understand their
        distributions.

    Use
    ---
    training_data_distributions(data)

    Required Packages
    -----------------
    pandas as pd
    matplotlib.pyplot as plt
    """
    # Determine the dimensions of the plot #
    n_row = int(len(channel_names) / 3) # 3 plots per row

    # Set the plot iteration value #
    plot_num = 1

    # Iterate through each subplot row and add the plots #
    for channel in channel_names:
        plt.subplot(n_row,3,plot_num)
        plt.hist(train_data[channel][train_data.Label_Codes == 2],
                 color = 'black',
                 alpha = alpha,
                 bins = 50)
        plt.hist(train_data[channel][train_data.Label_Codes == 1],
                 color = 'blue',
                 alpha = alpha,
                 bins = 50)
        plt.hist(train_data[channel][train_data.Label_Codes == 0],
                 color = 'yellow',
                 alpha = alpha,
                 bins = 50)
        plt.title(channel + ' Distribution')

        plot_num += 1

    # Display the plot #
    plt.tight_layout()
    plt.show()

def training_data_combinations(train_data,
                               channel_names = [('Red', 'Green', 'Blue'),
                                                ('Hue', 'Saturation', 'Value'),
                                                ('Luminosity', 'A_Axis', 'B_Axis')],
                               colors = {'Aleurone' : 'yellow',
                                         'Pericarp' : 'blue',
                                         'Light_Pericarp' : 'green',
                                         'Dark_Pericarp' : 'black',
                                         'Background' : 'purple'},
                               alpha = 0.2):
    """
    Parameters
    ----------
    train_data : a pandas dataframe to be used in cross validated machine
        learning. Should not contain test data.
    channel_names : a [list] of all possible columns to use
    colors : a {dictionary} of colors to use for each label
    alpha : how transparent to make the images

    Returns
    -------
    A plot of linear correlations between columns in the training data

    Use
    ---
    training_data_combinations(data)

    Required Packages
    -----------------
    pandas as pd
    matplotlib.pyplot as plt
    """
    # Determine the number of needed rows #
    n_row = int(len(channel_names))

    # Set the plot iteration value #
    plot_num = 1

    # Iterate through each subplot to create the plot #
    for channel_group in channel_names:
        channel_num_1 = 0
        channel_num_2 = 1

        for channel in channel_group:
            plt.subplot(n_row,3,plot_num)
            plt.scatter(train_data[channel_group[channel_num_1]],
                        train_data[channel_group[channel_num_2]],
                        c = train_data.Label.map(colors),
                        alpha = alpha,
                        marker = '.')
            plt.xlabel(channel_group[channel_num_1])
            plt.ylabel(channel_group[channel_num_2])

            plot_num += 1
            channel_num_1 += 1
            channel_num_2 += 1

            if channel_num_1 == 2:
                channel_num_2 = 0

    # Display the plot #
    plt.tight_layout()
    plt.show()

def training_data_correlations(train_data, show_num = False):
    """
    Parameters
    ----------
    train_data : a pandas dataframe to be used in cross validated machine
        learning. Should not contain test data.
    show_num : a constant to determine if PVE should be printed to screen

    Returns
    -------
    A heatmap of correlations between variables.
    Optional: a print of what the PVE is between each set of variables
        (pearson R^2).

    Use
    ---
    training_data_correlations(data)
    training_data_correlations(data, show_num = True)

    Required Packages
    -----------------
    pandas as pd
    matplotlib.pyplot as plt
    """
    # Create the heatmap of pearson R^2 values #
    plt.imshow(train_data.corr() ** 2, cmap = 'hot', interpolation = 'nearest')
    plt.xticks(ticks = range(0,10),
               labels = train_data.columns.values.tolist()[1:],
               rotation = -45,
               ha = 'left',
               rotation_mode = 'anchor')
    plt.yticks(ticks = range(0,10),
               labels = train_data.columns.values.tolist()[1:])
    plt.colorbar()
    plt.title('PVE Between Features and Label')
    plt.show()

    # If asked for, give the PVE values associated with the plot above #
    if show_num is True:
        print('Training Data PVE:\n', train_data.corr() ** 2)

def train_many_models(train_data,
                     models_list = [('NB',
                                     naive_bayes.GaussianNB(),
                                     {}),
                                    ('KNN',
                                     neighbors.KNeighborsClassifier(),
                                     {'n_neighbors' : (3, 5, 7, 9),
                                      'weights' : ('uniform',
                                                   'distance')}),
                                    ('DT',
                                     tree.DecisionTreeClassifier(),
                                     {'criterion' : ('gini',
                                                     'entropy')}),
                                    #('LOGIT',
                                    # linear_model.LogisticRegression(),
                                    # {'penalty' : ('l1',
                                    #               'l2'),
                                    #  'C' : (1, 10, 50, 100),
                                    #  'fit_intercept' : (True,
                                    #                     False)}),
                                    ('SVC',
                                     svm.SVC(),
                                     {'C' : (1, 10, 50, 100, 500, 1000),
                                      'kernel' : ('linear',
                                                  'rbf',
                                                  'sigmoid')}),
                                    ('RF',
                                     ensemble.RandomForestClassifier(),
                                     {'criterion' : ('gini',
                                                     'entropy'),
                                      'n_estimators' : (3, 25, 50, 75, 100)}),
                                    #('XGB',
                                     #xgb.XGBClassifier(eval_metric = 'mlogloss',
                                     #                         use_label_encoder = False),
                                     #{'learning_rate' : (0.1, 0.3, 0.5, 0.7, 0.9),
                                     # 'booster' : ('gbtree',
                                     #              'gblinear',
                                     #              'dart')}),
                                    ('RC',
                                     linear_model.RidgeClassifier(),
                                     {'alpha' : (0.5, 1, 1.5, 2, 3, 5),
                                      'fit_intercept' : (True,
                                                         False),
                                      'normalize' : (True,
                                                     False)})],
                     features = ['Red', 'Green', 'Blue',
                                'Hue', 'Saturation', 'Value',
                                'Luminosity', 'A_Axis', 'B_Axis']):
    """
    Parameters
    ----------
    train_data : a pandas dataframe to be used in cross validated machine
        learning. Should not contain test data.
    models_list : a [list] of models with dictionaries for hyperparameter
        values.  A default list is present for early exploration.
    features : a [list] of features to be used in the training of the model.
        Should be a list of column names.

    Returns
    -------
    best_models : a list of the best set of hyperparameters for each model as
        well as some performance statistics for each.

    Use
    ---
    best_models = train_many_models(data)

    Required Packages
    -----------------
    sklearn (as well as the model packages)
    pandas as pd
    """

    # Separating Training Data #
    train_x = train_data[features]
    train_y = train_data['Label_Codes']

    # Storage Lists #
    best_models = {}

    # For each model, figure out the best set of hyperparameters #
    for _, model, params in models_list:
        seed(7)
        start = time.time()
        grid_cv = GridSearchCV(estimator = model,
                               cv = 10,
                               param_grid = params,
                               scoring = 'accuracy')
        grid_cv.fit(train_x,
                    train_y)
        end = time.time()

        # If the model is not already in the list, add it #
        if grid_cv.best_estimator_ not in best_models:
            best_models[grid_cv.best_estimator_] = {'Time' : end - start,
                                                    'Avg_Accuracy' : grid_cv.cv_results_['mean_test_score'][grid_cv.best_index_],
                                                    'Var_Accuracy' : grid_cv.cv_results_['std_test_score'][grid_cv.best_index_]}

    # Return the list of best models and performances #
    return best_models

def learning_curves(train_data,
                    model,
                    features = ['Red', 'Green', 'Blue',
                               'Hue', 'Saturation', 'Value',
                               'Luminosity', 'A_Axis', 'B_Axis'],
                    train_sizes = np.linspace(0.005,1.0,20),
                    n_cv = 10):
    """
    Parameters
    ----------
    train_data : a pandas dataframe to be used in cross validated machine
        learning. Should not contain test data, a subsample will be kept aside
        for testing during the creation of the learning curve.
    model : a chosen model from sklearn to train
    features : A list of features to include in training.
        Should be a [list] of column names
    train_sizes : a np.linspace of all proportions to try.
    n_cv : an integer of the number of cross validations to perform.

    Returns
    -------
    Tuple of size 4. Contains the training sizes, training scores, testing
        scores, and fit times

    Use
    ---
    train_sizes, train_scores, test_scores, fit_times = learning_cuves(data,
                                                                       model)

    Required Packages
    -----------------
    sklearn (including the model package)
    random
    pandas as pd
    """
    # Separating Training Data #
    train_x = train_data[features]
    train_y = train_data['Label_Codes']

    # Set Seed for Consistency #
    seed(7)

    # Create Learning Curve Data #
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model,
                                                                          train_x,
                                                                          train_y,
                                                                          train_sizes = train_sizes,
                                                                          cv = n_cv,
                                                                          return_times = True)

    # Return Learning Curve Data #
    return train_sizes, train_scores, test_scores, fit_times
