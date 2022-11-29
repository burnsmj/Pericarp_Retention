### Opening Images in Python
### Michael Burns
### 9/10/21

"""
Purpose: To learn how to use python
         To learn how to open images in python with skimage
         To learn how to extract pixel information from the image
         To learn how to normalize an image
         To learn how to perform unsupervised learning (kmeans clustering)
         To learn how to perform PCA in python
         To understand what PCA does to RGB data
         To understand the use of image normalization
         To determine the best variables for clustering analysis
"""

#######################
# Importing Libraries #
#######################
import numpy as np # Numerical Python
import pandas as pd # Data wrangling
from skimage import color
from skimage import io
from sklearn.cluster import KMeans # Clustering analysis
from sklearn.decomposition import PCA # Principal Component Analysis
import matplotlib.pyplot as plt # Plotting
import matplotlib as mpl

########################
# Plotting Adjustments #
########################
mpl.rcParams['figure.dpi'] = 300 # Changing default figure resolution (increase)

##################
# Read RGB Image #
##################
image_rgb = io.imread('../Data/Images/Trials/Poststain_White_MB_H2O.png')[:,:,0:3]


def image_normalization (image):
    """

    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGBA photo, with a scale of 0-1 rather than 0-255.

    Returns
    -------
    normalized_image : NumPy array of a PNG photo
        Has the same dimensions as the input image, but each R,G, and B value
        is divided by the sum of the R, G, and B values of that pixel.

    Use
    ---
    image_normalization(<image_name>) : Returns a NumPy array stack (NxMx3)
    where NxM is the number of rows and columns of the original image, and 3 is
    the number of stacks (normalized red, normalized green, and normalized blue)
    plt.imshow(image_normalization(<image_name>)) : Returns a pyplot image of
    the normalized image

    Required Packages
    -----------------
    numpy as np

    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    ########################
    # Extract RGB Channels #
    ########################
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    ###############################
    # Normalize Pixel Intensities #
    ###############################
    tot_intensity = red + green + blue # Get sum of pixel intensities

    norm_red = np.nan_to_num(red / tot_intensity) * 255
    norm_green = np.nan_to_num(green / tot_intensity) * 255
    norm_blue = np.nan_to_num(blue / tot_intensity) * 255

    #########################################
    # Convert Normalized Values to Integers #
    #########################################
    norm_red = norm_red.astype(int) # Convert pixel values to integers for image plotting
    norm_green = norm_green.astype(int) # Convert pixel values to integers for image plotting
    norm_blue = norm_blue.astype(int) # Convert pixel values to integers for image plotting

    print('Pixels Normalized')

    ###########################
    # Stack Normalized Arrays #
    ###########################
    normalized_image = np.stack((norm_red, norm_green, norm_blue), axis = 2)

    assert(normalized_image.shape[0:2] == image.shape[0:2]), \
        'Normalized image is not the same size as the original'
    assert(isinstance(normalized_image, (np.ndarray, np.generic))), \
        'Normalized image is not a numpy array'

    ######################
    # Return Information #
    ######################
    print('Created normalized image')
    return normalized_image



def plot_pixel_distribution (image, max_count = 100000):
    """

    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGBA photo, with a scale of 0-1 rather than 0-255.

    Returns
    -------
    Plot of pixel distributions from 0 to 255.

    Use
    ---
    Plot_Pixel_Distribution(<image_name>) : Returns a 2x3 pyplot image of
    labelled histograms

    Required Packages
    -----------------
    numpy as np
    matplotlib.pyplot as plt

    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    ########################
    # Convert Image to HSV #
    ########################
    image_hsv = color.rgb2hsv(image)

    ##################################
    # Extract Channels and Normalize #
    ##################################
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]
    hue = image_hsv[:,:,0]
    sat = image_hsv[:,:,1]
    val = image_hsv[:,:,2]

    ####################################
    # Plot Pixel Intensity Information #
    ####################################
    plt.subplot(2,3,1)
    plt.hist(red.ravel(), bins = 256)
    plt.title('Red')
    plt.ylim(0,max_count)
    plt.ticklabel_format(style = 'plain')
    plt.subplot(2,3,2)
    plt.hist(green.ravel(), bins = 256)
    plt.title('Green')
    plt.ylim(0,max_count)
    plt.ticklabel_format(style = 'plain')
    plt.subplot(2,3,3)
    plt.hist(blue.ravel(), bins = 256)
    plt.title('Blue')
    plt.ylim(0,max_count)
    plt.ticklabel_format(style = 'plain')
    plt.subplot(2,3,4)
    plt.hist(hue.ravel(), bins = 256)
    plt.title('Hue')
    plt.ylim(0,max_count)
    plt.ticklabel_format(style = 'plain')
    plt.subplot(2,3,5)
    plt.hist(sat.ravel(), bins = 256)
    plt.title('Saturation')
    plt.ylim(0,max_count)
    plt.ticklabel_format(style = 'plain')
    plt.subplot(2,3,6)
    plt.hist(val.ravel(), bins = 256)
    plt.title('Value')
    plt.ylim(0,max_count)
    plt.ticklabel_format(style = 'plain')
    plt.tight_layout()
    plt.show()

    ######################
    # Return Information #
    ######################
    print('Plotted pixel distributions')



def tabulate_pixels (image):
    """

    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGBA photo, with a scale of 0-1 rather than 0-255.

    Returns
    -------
    Pixel_Data : A pandas dataframe of all channels in a long format. This will
    return Red, Green, and Blue columns, even if normalized data is used.

    Use
    ---
    tabulate_pixels(<image_name>) : Returns a long dataframe of every pixel
    channel written out

    Required Packages
    -----------------
    numpy as np
    pandas as pd

    """


    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    ##################################
    # Extract Channels and Normalize #
    ##################################
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    ##########################################
    # Create Data Frame of Pixel Information #
    ##########################################
    pixel_data = pd.DataFrame({'Red' : red.ravel(),
                               'Green' : green.ravel(),
                               'Blue' : blue.ravel()})

    ######################
    # Return Information #
    ######################
    print('Tabulated Pixel Data')
    return pixel_data



def pixel_pca (image):
    """

    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGBA photo, with a scale of 0-1 rather than 0-255.

    Returns
    -------
    pca_data : A pandas dataframe of the first two principle components of
    every pixel

    Use
    ---
    pixel_pca(<image_name>) : Returns a long dataframe of the first two
    principal components for each pixel

    Required Packages
    -----------------
    numpy as np
    pandas as pd
    PCA from sklearn.decomposition

    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    #######################
    # Tabulate Image Data #
    #######################
    pixel_data = tabulate_pixels(image)

    #############################
    # Perform PCA on Image Data #
    #############################
    pca = PCA(2)
    pca_data = pca.fit_transform(pixel_data[['Red', 'Green', 'Blue']])
    pca_df = pd.DataFrame({'PC1' : pca_data[:,0],
                             'PC2' : pca_data[:,1]})

    ######################
    # Return Information #
    ######################
    print('Performed PCA')
    return pca_df



def pixel_clustering(image, clusters = 3, use_pca = False):
    """

    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGBA photo, with a scale of 0-1 rather than 0-255.
    clusters : Integer (default = 3)
        Should be an integer of the number of clusters to use in k-means clustering
    PCA : Boolean
        If True, PCA will be performed on the image before clustering
        If False, the image will be processed as is.

    Returns
    -------
    Wide_Predictions : A wide dataset of cluster classifications for each pixel
    based either on RGB data or PCA data

    Use
    ---
    Pixel_Clustering(<image_name>) : Returns a wide dataframe (same dimensions
    as the original image) of the kmeans clustering predictions of every pixel
    Pixel_Clustering(<image_name>, clusters = 5) : Returns a wide dataframe
    (same dimension as the original image), of the kmeans clustering prediction
    of every pixel with 5 classes
    Pixel_Clustering(<image_name>, PCA = True) : Returns a wide dataframe
    (same dimensions as the original image) of the kmeans clustering predictions
    of every pixel after extracting the first two principal components of the pixel

    Required Packages
    -----------------
    numpy as np
    pandas as pd
    PCA from sklearn.decomposition
    KMeans from sklearn.cluster

    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    ###################################
    # Create K Means Clustering Model #
    ###################################
    kmeans_model = KMeans(n_clusters = clusters, random_state = 7)

    ############################################################################
    # Determine if Data Should be RGB or PCA and Perform Appropriate Functions #
    ############################################################################
    if use_pca is True:
        pixel_data = pixel_pca(image)
        kmeans_model.fit(pixel_data[['PC1', 'PC2']])
        predictions = kmeans_model.predict(pixel_data[['PC1', 'PC2']])
    elif use_pca is False:
        pixel_data = tabulate_pixels(image)
        kmeans_model.fit(pixel_data[['Red','Green','Blue']])
        predictions = kmeans_model.predict(pixel_data[['Red','Green','Blue']])
    else:
        print('Warning: PCA argument must be set to True or False')

    #########################################################
    # Pivot Normalized Pixel Prediction Data to Wide Format #
    #########################################################
    wide_predictions = np.array(predictions).reshape(image.shape[0], image.shape[1])

    assert(wide_predictions.shape == image.shape[0:2]), \
        'Predicted image is not the same size as the original'
    assert(isinstance(wide_predictions, (np.ndarray, np.generic))), \
        'Predicted image is not a numpy array'

    ######################
    # Return Information #
    ######################
    print('Clustering Complete')
    return wide_predictions



def plotting_cluster_images(image):
    """

    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGBA photo, with a scale of 0-1 rather than 0-255.

    Returns
    -------
    2x3 plot of original image, predicted clusters of original image, predicted
    clsuters of PCA of original image
                normalized image, predicted clusters of normalized image,
                predicted clsuters of PCA of normalized image

    Use
    ---
    Plotting_Cluster_Images(<image_name>) : Returns the 2x3 plot described above

    Required Packages
    -----------------
    numpy as np
    pandas as pd
    PCA from sklearn.decomposition
    KMeans from sklearn.cluster
    matplotlib.pyplot as plt

    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    ################################
    # Plot All Images Side-by-Side #
    ################################
    plt.subplot(2,3,1)
    plt.imshow(image)
    plt.ylabel('Basic Image')
    plt.title('Image', fontdict={'fontsize' : 10})
    plt.tick_params(left = False,
                    bottom = False,
                    right = False,
                    labelleft = False,
                    labelbottom = False)
    plt.subplot(2,3,2)
    plt.imshow(pixel_clustering(image))
    plt.title('Pixel\nPredictions', fontdict={'fontsize' : 10})
    plt.tick_params(left = False,
                    bottom = False,
                    right = False,
                    labelleft = False,
                    labelbottom = False)
    plt.subplot(2,3,3)
    plt.imshow(pixel_clustering(image, use_pca = True))
    plt.title('PCA\nPredictions', fontdict={'fontsize' : 10})
    plt.tick_params(left = False,
                    bottom = False,
                    right = False,
                    labelleft = False,
                    labelbottom = False)
    plt.subplot(2,3,4)
    plt.imshow(image_normalization(image))
    plt.ylabel('Normalized Image')
    plt.tick_params(left = False,
                    bottom = False,
                    right = False,
                    labelleft = False,
                    labelbottom = False)
    plt.subplot(2,3,5)
    plt.imshow(pixel_clustering(image_normalization(image)))
    plt.tick_params(left = False,
                    bottom = False,
                    right = False,
                    labelleft = False,
                    labelbottom = False)
    plt.subplot(2,3,6)
    plt.imshow(pixel_clustering(image_normalization(image), use_pca = True))
    plt.tick_params(left = False,
                    bottom = False,
                    right = False,
                    labelleft = False,
                    labelbottom = False)
    plt.subplots_adjust(wspace = -0.7, hspace = 0.05)
    plt.show()
    #plt.savefig(filename, dpi = 300, bbox_inches = 'tight')

    ######################
    # Return Information #
    ######################
    print('Plotting Complete')



def plotting_differential_clusters (image, min_cluster = 2, max_cluster = 8):
    """

    Parameters
    ----------
    image : NumPy array of a PNG photo
        Should be an RGBA photo, with a scale of 0-1 rather than 0-255.
    min_clusters : Integer
        Should be an integer value for the minimum number of cluster to include
        in the plotting for analysis
    max_cluster : Integer
        Should be an integer value for the maximum number of clusters to
        include in the plotting for analysis

    Returns
    -------
    2xM plot of original image (number of columns is determined by the number
                                of different clusters to be used) in the
    following order:
        original image, image using minimum number of clusters (2 by default),
        ..., image using maximum number of clusters (8 by default)

    Use
    ---
    Plotting_Differential_Clusters(<image_name>) : Returns a 2x4 plot described
    above
    Plotting_Differential_Clusters(<image_name>, min_cluster = 4) : Returns a
    2x3 plot described above

    Required Packages
    -----------------
    numpy as np
    pandas as pd
    PCA from sklearn.decomposition
    KMeans from sklearn.cluster
    matplotlib.pyplot as plt

    """
    ###########################
    # Check Image Information #
    ###########################
    assert(isinstance(image, (np.ndarray, np.generic))), \
        "Image is not loaded as a numpy array"

    subplot_columns = int((max_cluster - min_cluster + 2) / 2)
    print('Number of Columns:', subplot_columns)

    first_img_pos = 1

    plt.subplot(2,subplot_columns,1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    for cluster in range(min_cluster, max_cluster+1):
        first_img_pos += 1
        plt.subplot(2,subplot_columns,first_img_pos)
        plt.imshow(pixel_clustering(image, clusters = cluster))
        plt.axis('off')
        plt.title('Clusters: ' + str(cluster))

    plt.tight_layout()
    plt.show()

    ######################
    # Return Information #
    ######################
    print('Plotting Complete')
    
    
def inidiv_cluster_plotting(image, clusters = 3):
    img_pos = 1
    
    image_orig = image.copy()
    
    plt.subplot(1,clusters+1,1)
    plt.imshow(image_orig)
    plt.title('Original')
    plt.axis('off')

    img_pos += 1

    cluster_data = pixel_clustering(image, clusters = clusters)
    image_cp = image.copy()

    for cluster in range(0, clusters):
        image[cluster_data != cluster] = 0

        plt.subplot(1,clusters+1,img_pos)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Cluster: ' + str(cluster))

        img_pos += 1
        
        image = image_cp.copy()

    plt.tight_layout()
    plt.show()

# NOT A VIABLE OPTION FOR ANNOTATION.  LEAVES TOO MANY PIXELS ON LIGHTLY COLORED PERICARP
#images = sorted(glob.glob('*.tif'))
#for image in images:
#    print(image)
#    test_image = io.imread(image, plugin = 'pil')
#    
#    for i in range(5,8):
#        inidiv_cluster_plotting(test_image.copy(), clusters = i)



#plt.subplot(1,4,1)
#plt.imshow(image_rgb)
#plt.axis('off')
#plt.subplot(1,4,2)
#plt.imshow(red)
#plt.axis('off')
#plt.subplot(1,4,3)
#plt.imshow(green)
#plt.axis('off')
#plt.subplot(1,4,4)
#plt.imshow(blue)
#plt.axis('off')
#plt.tight_layout()
#plt.show()
