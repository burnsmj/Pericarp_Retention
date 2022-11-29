#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:38:11 2022

@author: michael
"""
### Color Intensity of Pericarp POC
### Michael Burns
### 8/23/2022

# Packages
from lxml import etree
from glob import glob
from random import sample
import numpy as np
import pandas as pd
from skimage import draw
from skimage import io
from skimage import color
import matplotlib.pyplot as plt

# Options
pd.set_option('display.max_columns', 15)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 25
plt.rcParams["figure.figsize"] = (8,5)

# Functions
def xml_polygon_drawing(xml_tree_element, annotation_shape = 'polygon'):
    """
    Parameters
    ----------
    xml_tree : XML File
        An XML file with children labelled 'image' with subelements of 'name', 'width', 'height', and 'points'.
    shape : 'polygon' or 'box'
        Informs on which method to use when interpreting the xml file.
    show_images : Boolean, optional
        Whether or not to show the image overlayed by the mask. The default is False.

    Returns
    -------
    Mask image with highlighted area from within the polygon.
    """
    
    if annotation_shape == 'polygon':
        #####################################################
        # Fill the Labelled Mask Based on Polygon Verticies #
        #####################################################
        for child in xml_tree_element:
            points = child.attrib['points'].split(';') # Extract verticies
    
            points_x = [] # Initialize a list of x coordinates
            points_y = [] # Initialize a list of y coordinates
    
            ############################################################
            # Extract the X and Y Coordinates of the Polygon Verticies #
            ############################################################
            for point in points:
                points_x.append(round(float(point.split(',')[1]) - 1)) # Extract the x coordinate.  The '-1' is for indexing since python is base-zero
                points_y.append(round(float(point.split(',')[0]) - 1)) # Extract the y coordinate.  The '-1' is for indexing since python is base-zero
    
            polygon_x, polygon_y = draw.polygon(points_x, points_y) # Create a polygon shape.
    
            mask[polygon_x, polygon_y] = 2 # Color the mask by the polygon
    elif annotation_shape == 'box':
        mask[mask == 0] = 2 # Color the mask by the polygon, since it is set to square, the whole kernel must be covered in pericarp
    else:
        print('Shape is not either polygon or box')

    ##########################
    # Return the Image Masks #
    ##########################
    return mask


# Read in xml file
#xml_tree = etree.parse("../Data/Annotation_Files/Pericarp_Annotations_Round1.xml")
#xml_tree = etree.parse("/Users/michael/Downloads/annotations_2.xml")
xml_tree = etree.parse("/Users/michael/Downloads/annotations_3.xml")
images = sorted(glob('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Annotation_Images/First_Round_Split_Images/*/*.tif'))
germ_dir = pd.read_excel('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Germ_Directions.xlsx')
images_geno = [i.split('/')[-1] for i in images]

# Hard code in the annotators and the task id codes for each of them to separate later.  Had to be hard coded because lxml does not easily give up strings.
task_id_codes = [196510, 196524, 196528, 213283, 213287, 219280, 219281, 219285, 219288, 219290, 224283]
annotators = ['berry603', 'helle426', 'joh20447', 'berry603', 'joh20447', 'joh20447', 'berry603', 'berry603', 'helle426', 'helle426', 'lovel144']

# Initiate a dictionary to add pericarp coverage to
pericarp_chan1_data = []
pericarp_chan2_data = []
pericarp_chan3_data = []
aleurone_chan1_data = []
aleurone_chan2_data = []
aleurone_chan3_data = []

tasks = []

show_images = False

iter = 0
for elem in xml_tree.getroot().iter('image'):
    if iter % 100 == 0:
        print('Working on iterations ' + str(iter) + ' through ' + str(iter + 99) + ' of 2299')

    iter += 1
    
    #############################
    # Extract Image Information #
    #############################
    fname = elem.attrib['name'] # Get the file name
    fwidth = int(elem.attrib['width']) # Get the pixel width
    fheight = int(elem.attrib['height']) # Get the pixel height
    task_id = int(elem.attrib['task_id'])
    image_id = int(elem.attrib['id'])
    #print(fname) # Debugging
    
    # Skip if the germ is visible(Y)/invisible(N)
    # Increases ordinary correlations when looking at germ side
    # Increases correlation without outliers when not looking at germ side
    #germ_vis = germ_dir[germ_dir.Image_ID == image_id].Germ_Visible

    #if list(germ_vis)[0] == 'Y':
     #   continue

    if 'polygon' in str(elem[0]):
        shape = 'polygon'
    elif 'box' in str(elem[0]):
        shape = 'box'
    else:
        print('Shape was not found as a polygon or box: ' + fname)
        continue
        
    # Check fname since duplicates have an extra _N added to their file name
    if len(fname.split('_')) == 4:
        fname_new = fname[0:-6] + '.tif'
    else:
        fname_new = fname
    
    ##########################
    # Read in Original Image #
    ##########################
    orig_img = io.imread(images[images_geno.index(fname_new)], plugin = 'pil') # Read in the image
    
    ########################
    # Create Mask of Image #
    ########################
    mask = np.zeros([fheight, fwidth])
    
    annotated_mask = xml_polygon_drawing(elem, annotation_shape = shape)   
    annotated_mask[(orig_img == [0,0,0]).all(axis = 2)] = 0 # List anything that was initially background as background. Annotation erred on the side of selecting background as pericarp becuase background is clearly marked as [0,0,0] in the images
    annotated_mask[np.logical_and((orig_img != [0,0,0]).all(axis = 2), mask != 2)] = 1 # Label the aleurone as any area that isn't background in the original image, and wasn't annotated as pericarp

    ##################################
    # Determine Color Space of Image #
    ##################################
    image_colored = color.rgb2lab(orig_img)

    ###################
    # Append New Data #
    ###################
    pericarp_chan1_data += list(image_colored[:,:,0][annotated_mask == 2])
    pericarp_chan2_data += list(image_colored[:,:,1][annotated_mask == 2])
    pericarp_chan3_data += list(image_colored[:,:,2][annotated_mask == 2])
    aleurone_chan1_data += list(image_colored[:,:,0][annotated_mask == 1])
    aleurone_chan2_data += list(image_colored[:,:,1][annotated_mask == 1])
    aleurone_chan3_data += list(image_colored[:,:,2][annotated_mask == 1])
    tasks.append(task_id)

plt.subplot(3,1,1)
plt.hist(sample(pericarp_chan1_data, 10000), color = 'blue', alpha = 0.5, bins = 255)
plt.hist(sample(aleurone_chan1_data, 10000), color = 'orange', alpha = 0.5, bins = 255)
plt.xticks([])
plt.subplot(3,1,2)
plt.hist(sample(pericarp_chan2_data, 10000), color = 'blue', alpha = 0.5, bins = 255)
plt.hist(sample(aleurone_chan2_data, 10000), color = 'orange', alpha = 0.5, bins = 255)
plt.xticks([])
plt.subplot(3,1,3)
plt.hist(sample(pericarp_chan3_data, 10000), color = 'blue', alpha = 0.5, bins = 255)
plt.hist(sample(aleurone_chan3_data, 10000), color = 'orange', alpha = 0.5, bins = 255)
plt.show()










#images = sorted(glob('../Data/Images/POC/Color_Intensity/*.tif'))

#print(images)

#for image in images:
#    img_rgb = io.imread(image, plugin = 'pil')
#    img_lab = color.rgb2lab(img_rgb)
#    img_hsv = color.rgb2hsv(img_rgb)
    
#    plt.subplot(1,5,1)
#    plt.imshow(img_rgb)
#    plt.axis('off')
#    plt.subplot(1,5,2)
#    plt.imshow(img_rgb[:,:,2], cmap = 'gray')
#    plt.axis('off')
#    plt.subplot(1,5,3)
#    plt.imshow(img_lab[:,:,2], cmap = 'gray')
#    plt.axis('off')
#    plt.subplot(1,5,4)
#    plt.imshow(img_hsv[:,:,1], cmap = 'gray')
#    plt.axis('off')
#    plt.subplot(1,5,5)
#    plt.imshow(-1 * (img_lab[:,:,2] - img_hsv[:,:,1]), cmap = 'gray')
#    plt.axis('off')
#    plt.show()

#plt.scatter(img_lab[:,:,0].ravel(), img_lab[:,:,2].ravel())
#plt.show()

#b_ravel = []
#for image in images:
#    img_lab = color.rgb2lab(io.imread(image, plugin = 'pil'))
#    img_hsv = color.rgb2hsv(io.imread(image, plugin = 'pil'))
#    
#    b_ravel += list((-1 * (img_hsv[:,:,1] - img_lab[:,:,2])).ravel())

#plt.hist(b_ravel, bins = 255)
#plt.ylim((0,20000))
#plt.show()
