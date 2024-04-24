#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 09:25:55 2022

@author: michael
"""
### Practice XML Parsing
### Michael Burns
### 2/2/22

###################
# Import Packages #
###################
from lxml import etree
import numpy as np
from skimage import draw
from skimage import io as io
import matplotlib.pyplot as plt

image = io.imread('../test_file_3.tif', plugin = 'pil')

#################
# Read XML File #
#################
tree = etree.parse("../../../../../../Downloads/annotations.xml")

fname = tree.getroot()[2].attrib['name']
fwidth = int(tree.getroot()[2].attrib['width'])
fheight = int(tree.getroot()[2].attrib['height'])

print(fname)
print(fwidth)
print(fheight)

print(image.shape)

points = tree.getroot()[2][0].attrib['points'].split(';')

points_x = []
points_y = []

for point in points:
    points_x.append(round(float(point.split(',')[0])))
    points_y.append(round(float(point.split(',')[1])))

print(points_x)
print(points_y)

mask = np.zeros([fheight, fwidth])

print(mask.shape)

polygon_x, polygon_y = draw.polygon(points_y, points_x)

mask[polygon_x, polygon_y] = 1
mask[(image == [0,0,0]).all(axis = 2)] = 0

plt.imshow(image)
plt.axis('off')
plt.show()

plt.imshow(image)
plt.imshow(mask, alpha = 0.7)
plt.axis('off')
plt.show()

def xml_polygon_drawing(xml_tree, output_path = '../Data/', show_images = False):
    """
    Parameters
    ----------
    xml_tree : XML File
        An XML file with children labelled 'image' with subelements of 'name', 'width', 'height', and 'points'.
    output_path : File Path, optional
        A file path to output mask files to. The default is '../Data/'.
    show_images : Boolean, optional
        Whether or not to show the image overlayed by the mask. The default is False.

    Returns
    -------
    Automatically saves mask files to the designated directory.
    """

    ####################################
    # Split Information for Each Image #
    ####################################
    for elem in xml_tree.getroot().iter('image'):
        fname = elem.attrib['name'] # Get the file name
        fwidth = int(elem.attrib['width']) # Get the pixel width
        fheight = int(elem.attrib['height']) # Get the pixel height

        mask = np.zeros([fheight, fwidth]) # Create a mask of zeros that is the same size as the image

        print(fname) # Print the name of the image that is being processed.
        orig_img = io.imread('../' + fname) # Read in the image

        #####################################################
        # Fill the Labelled Mask Based on Polygon Verticies #
        #####################################################
        for child in elem:
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

            mask[polygon_x, polygon_y] = 1 # Color the mask by the polygon

        mask[(orig_img == [0,0,0]).all(axis = 2)] = 0 # List anything that was initially background as background. Annotation erred on the side of selecting background as pericarp becuase background is clearly marked as [0,0,0] in the images
        mask[np.logical_and((orig_img != [0,0,0]).all(axis = 2), mask != 1)] = 2 # Label the aleurone as any area that isn't background in the original image, and wasn't annotated as pericarp

        #################################
        # Show Image Overlay if Desired #
        #################################
        if show_images is True:
            plt.subplot(1,3,1)
            plt.imshow(orig_img) # Plot the original image
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(mask, cmap = 'gray') # On top of that, show the mask (with transparency)
            plt.axis('off') # Don't display the axes
            plt.subplot(1,3,3)
            plt.imshow(orig_img)
            plt.imshow(mask, cmap = 'gray', alpha = 0.5)
            plt.axis('off')
            plt.show() # Show the plot

        #################################
        # Save the Labelled Mask Images #
        #################################
        np.savetxt(output_path + fname.split('.')[0] + '_mask.txt', mask.astype(int), fmt='%i')

xml_polygon_drawing(tree, show_images=True)


