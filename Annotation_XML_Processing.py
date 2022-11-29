#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:19:03 2022

@author: michael
"""
### Annotation XML Parsing
### Michael Burns
### 8/12/22

# Purpose:  The purpose of this script is to parse through the XML files
#           provided by CVAT for the images that the undergradutes annotated.
#           In theory I should be able to utilize the draw function from
#           skimage to create the polygon, subtract the pixels that are set to
#           0,0,0 (background) and determine the number of pixels present to
#           determine the non-pericarp area.  I should also then be able to
#           calculate pericarp area by using the remaining pixels minus
#           background and determine the ratio of pericarp to aleurone. This
#           can then be correlated to the ground truth values of pericarp
#           retention for the genotypes that are the same between the two
#           datasets (note that none of the samples that have been annotated
#           were part of the ground truth scanning, so they might not be ideal
#           but should provide a somewhat decent idea of the performance).

#           I will also need to test the correlation between annotators.

#           In theory I can create a model from these 2000-2500 images to
#           predict on the 1400 images from ground truth data collection to
#           correlate and see how well the model performs. In order to make
#           this work, I am going to need to use image modifications to
#           increase sample size.

# Anticipated Steps:
    # 1. Import packages (skimage, numpy, lxml, plt)
    # 2. Read in data line by line
    #    a. look for '<image' at the beginning of the line
    #    b. get the file name
    #    c. extract the genotype, date, and replicate
    #    d. get the polygon vertices
    # 3. Read in the image associated with this annotation
    # 4. Draw the polygon
    # 5. Calculate the number of pixels in the polygon (that aren't background)
    # 6. Calculate the number of pixels left in the image (that aren't background)
    # 7. Calculate the pericarp coverage

# Packages
from lxml import etree
from glob import glob
import numpy as np
import pandas as pd
from skimage import draw
from skimage import io
from skimage import color
from skimage import transform
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Options
pd.set_option('display.max_columns', 15)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (8,8)

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

def pericarp_intensity(original_rgb, mask, pericarp_label = 2):
    #img_lab = color.rgb2lab(original_rgb)
    #img_hsv = color.rgb2hsv(original_rgb)
    
    image_values = 255 - original_rgb[:,:,2]
    
    #image_values = -1 * (img_hsv[:,:,1] - img_lab[:,:,2])

    pericarp_values = image_values[mask == pericarp_label]

    mean_pericarp_value = np.median(pericarp_values)
        
    return mean_pericarp_value

# Read in xml file
#xml_tree = etree.parse("../Data/Annotation_Files/Pericarp_Annotations_Round1.xml")
#xml_tree = etree.parse("/Users/michael/Downloads/annotations_2.xml")
xml_tree = etree.parse("/Users/michael/Downloads/pericarp_annotations_final.xml")
images = sorted(glob('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Annotation_Images/First_Round_Split_Images/*/*.tif'))
germ_dir = pd.read_excel('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Germ_Directions.xlsx')
images_geno = [i.split('/')[-1] for i in images]

# Hard code in the annotators and the task id codes for each of them to separate later.  Had to be hard coded because lxml does not easily give up strings.
task_id_codes = [196510, 196524, 196528, 213283, 213287, 219280, 219281, 219285, 219288, 219290, 224283]
annotators = ['berry603', 'helle426', 'joh20447', 'berry603', 'joh20447', 'joh20447', 'berry603', 'berry603', 'helle426', 'helle426', 'lovel144']

# Initiate a dictionary to add pericarp coverage to
df_cols = ['fname', 'Genotype', 'Date', 'Image_Number', 'Annotator', 'Coverage', 'Intensity', 'Quantity']
df_data = []

tasks = []

show_images = False
save_images = False
save_masks = False

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
    if show_images is True:
        plt.subplot(1,4,1)
        plt.imshow(orig_img)
        plt.axis('off')
    
    ########################
    # Create Mask of Image #
    ########################
    mask = np.zeros([fheight, fwidth])
    #if show_images is True:
    #    plt.subplot(1,5,2)
    #    plt.imshow(mask, cmap = 'gray')
    #    plt.axis('off')
    
    annotated_mask = xml_polygon_drawing(elem, annotation_shape = shape)
    if show_images is True:
        plt.subplot(1,4,2)
        plt.imshow(orig_img)
        plt.imshow(annotated_mask, cmap = 'gray', alpha = 0.5)
        for child in elem:
            points = child.attrib['points'].split(';') # Extract verticies
    
            points_x = [] # Initialize a list of x coordinates
            points_y = [] # Initialize a list of y coordinates
    
            ############################################################
            # Extract the X and Y Coordinates of the Polygon Verticies #
            ############################################################
            for point in points:
                points_x.append(round(float(point.split(',')[1]) - 1)) # Extract the x coordinate.  The '-1' is for indexing since python is base-zero
                points_y.append(round(float(point.split(',')[0]) - 1))
                
            points_x.append(points_x[0])
            points_y.append(points_y[0])
            plt.plot(points_y, points_x, c = 'red', linewidth = 5)
        plt.axis('off')
    
    annotated_mask[(orig_img == [0,0,0]).all(axis = 2)] = 0 # List anything that was initially background as background. Annotation erred on the side of selecting background as pericarp becuase background is clearly marked as [0,0,0] in the images
    if show_images is True:
        plt.subplot(1,4,3)
        plt.imshow(orig_img)
        plt.imshow(annotated_mask, cmap = 'gray', alpha = 0.5)
        plt.axis('off')
    
    annotated_mask[np.logical_and((orig_img != [0,0,0]).all(axis = 2), annotated_mask != 2)] = 1 # Label the aleurone as any area that isn't background in the original image, and wasn't annotated as pericarp
    if show_images is True:
        #from matplotlib import colors
        #cmap = colors.ListedColormap(['black', 'gray'])
        
        plt.subplot(1,4,4)
        plt.imshow(annotated_mask, cmap = 'gray')
        plt.axis('off')
        plt.tight_layout(pad = 0, rect = (0,0,5,5))
        plt.show()
    
    prop_pericarp_coverage = (annotated_mask == 2).sum() / ((annotated_mask == 2).sum() + (annotated_mask == 1).sum())
    
    pericarp_thickness = pericarp_intensity(orig_img, annotated_mask)
    
    pericarp_quantity = prop_pericarp_coverage * pericarp_thickness
    
    df_data.append([fname_new, fname.split('_')[0], int(fname.split('_')[1]), int(fname.split('.')[0].split('_')[2]), annotators[task_id_codes.index(task_id)], prop_pericarp_coverage, pericarp_thickness, pericarp_quantity])
    
    tasks.append(task_id)
    
    # Debugging to stop early
    #if iter == 3:
    #    break

    # Save image and transformations
    if save_images is True:
        for flip in [True, False]:
            for rotation in [0,90,180,270]:
                image = transform.rotate(orig_img, rotation, resize = True)
                flipped = 'nf'
                if flip is True:
                    image = np.fliplr(image)
                    flipped = 'f'
                io.imsave('~/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Annotation_Images/Split_Images/' + fname[:-4] + '_' + str(rotation) + '_' + flipped + '.png', image)

    # Save mask and transformations
    if save_masks is True:
        for flip in [True, False]:
            for rotation in [0,90,180,270]:
                mask = transform.rotate(annotated_mask, rotation, resize = True)
                flipped = 'nf'
                if flip is True:
                    mask = np.fliplr(mask)
                    flipped = 'f'
                io.imsave('~/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Annotation_Masks/' + fname[:-4] + '_' + str(rotation) + '_' + flipped + '.png', mask)

pericarp_coverage = pd.DataFrame(df_data, columns = df_cols)

print(pericarp_coverage)

##################################
# Correlation Between Annotators #
##################################
duplicate_annotations = pericarp_coverage[pericarp_coverage[['Genotype', 'Date', 'Image_Number']].duplicated(keep = False)]

duplicates_wide = duplicate_annotations.pivot(index = ('Genotype', 'Date', 'Image_Number'), columns = 'Annotator', values = 'Coverage')

duplicates_wide.corr()

# All of the correlations between annotators is above 0.9, which is great to see!

###################################
# Averageing Coverage by Genotype #
###################################
sorted_coverage_df = pericarp_coverage.assign(key = pericarp_coverage.Annotator.map({'joh20447' : 0,
                                                                                     'berry603' : 1,
                                                                                     'lovel144' : 2,
                                                                                     'helle426' : 3})).sort_values('key').drop('key', axis = 1)

deduplicated_df = sorted_coverage_df.drop_duplicates(subset = ('Genotype', 'Date', 'Image_Number'), keep = 'first') # This should reduce the data down the unique annotations, and the duplicates that joh20447 did.

deduplicated_df.Coverage.median()

averaged_coverage = deduplicated_df.groupby('Genotype')['Coverage'].median('Coverage').rename_axis('Sample_ID').reset_index()
averaged_intensity = deduplicated_df.groupby('Genotype')['Intensity'].median('Intensity').rename_axis('Sample_ID').reset_index()
averaged_quantity = deduplicated_df.groupby('Genotype')['Quantity'].median('Quantity').rename_axis('Sample_ID').reset_index()

# Plot distribution of average coverage
plt.hist((averaged_coverage.Coverage))
plt.show()

plt.hist((averaged_intensity.Intensity))
plt.show()

plt.hist((averaged_quantity.Quantity))
plt.show()

# This is very heavily weighted on the high end.  I need to make sure the averages are happening correctly.
########################################################
# Correlation Between Coverage and Proportion Retained #
########################################################
# Read in ground truth data and kernel mass data
cooked_data = pd.read_excel('../Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Rapid_Cooked')
kernel_masses = pd.read_excel('../Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Kernel_Mass')
imbibed_data = pd.read_excel('../Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Imbibed')

# Filter for non-stained samples
cooked_data = cooked_data[cooked_data.Stained == 'N']

# Select the columns that are needed
cooked_data = cooked_data[['Sample_ID', 'Average_Pericarp_Retained']]
kernel_masses = kernel_masses[['Sample_ID', 'Average_Kernel_Mass']]
imbibed_data = imbibed_data[['Sample_ID', 'Average_Pericarp_Initial']]

# Combine datasets
combined_data = cooked_data.merge(kernel_masses,
                         on = 'Sample_ID',
                         how = 'left').merge(imbibed_data,
                                             on = 'Sample_ID',
                                             how = 'left')

# Determine the normalized proportion of pericarp retained
combined_data['Normalized_Pericarp_Retained'] = combined_data.Average_Pericarp_Retained / combined_data.Average_Kernel_Mass
combined_data['Proportion_Pericarp_Retained'] = combined_data.Average_Pericarp_Retained / combined_data.Average_Pericarp_Initial
combined_data['Norm_Prop_Pericarp_Retained'] = combined_data.Proportion_Pericarp_Retained / combined_data.Average_Kernel_Mass
combined_data['Prop_Pericarp_Retained_Norm_Initial'] = combined_data.Proportion_Pericarp_Retained / (combined_data.Average_Pericarp_Initial / combined_data.Average_Kernel_Mass)

# Merge ground truth data with cooked data
combined_data = combined_data.merge(averaged_coverage,
                                    on = 'Sample_ID',
                                    how = 'inner').merge(averaged_intensity,
                                                         on = 'Sample_ID',
                                                         how = 'inner').merge(averaged_quantity,
                                                                              on = 'Sample_ID',
                                                                              how = 'inner')

# Determine correlations between percent coverage, average pericarp retained, and normalized pericarp retained
print('R of Pericarp Retained and Pericarp Coverage: ' + str(round(pearsonr(combined_data.Average_Pericarp_Retained, combined_data.Coverage)[0], 3)))
print('R of Normalized Pericarp Retained and Pericarp Coverage: ' + str(round(pearsonr(combined_data.Normalized_Pericarp_Retained, combined_data.Coverage)[0], 3)))
print('R of Proportion Pericarp Retained and Pericarp Coverage: ' + str(round(pearsonr(combined_data.Proportion_Pericarp_Retained, combined_data.Coverage)[0], 3)))
print('R of Normalized Pericarp Retained and Pericarp Coverage: ' + str(round(pearsonr(combined_data.Norm_Prop_Pericarp_Retained, combined_data.Coverage)[0], 3)))
print('R of Pericarp Retained per Normalized Initial Content and Pericarp Coverage: ' + str(round(pearsonr(combined_data.Prop_Pericarp_Retained_Norm_Initial, combined_data.Coverage)[0], 3)))

print('R of Pericarp Retained and Pericarp Intensity: ' + str(round(pearsonr(combined_data.Average_Pericarp_Retained, combined_data.Intensity)[0], 3)))
print('R of Normalized Pericarp Retained and Pericarp Intensity: ' + str(round(pearsonr(combined_data.Normalized_Pericarp_Retained, combined_data.Intensity)[0], 3)))
print('R of Proportion Pericarp Retained and Pericarp Intensity: ' + str(round(pearsonr(combined_data.Proportion_Pericarp_Retained, combined_data.Intensity)[0], 3)))
print('R of Normalized Pericarp Retained and Pericarp Intensity: ' + str(round(pearsonr(combined_data.Norm_Prop_Pericarp_Retained, combined_data.Intensity)[0], 3)))
print('R of Pericarp Retained per Normalized Initial Content and Pericarp Intensity: ' + str(round(pearsonr(combined_data.Prop_Pericarp_Retained_Norm_Initial, combined_data.Intensity)[0], 3)))

print('R of Pericarp Retained and Pericarp Quantity: ' + str(round(pearsonr(combined_data.Average_Pericarp_Retained, combined_data.Quantity)[0], 3)))
print('R of Normalized Pericarp Retained and Pericarp Quantity: ' + str(round(pearsonr(combined_data.Normalized_Pericarp_Retained, combined_data.Quantity)[0], 3)))
print('R of Proportion Pericarp Retained and Pericarp Quantity: ' + str(round(pearsonr(combined_data.Proportion_Pericarp_Retained, combined_data.Quantity)[0], 3)))
print('R of Normalized Pericarp Retained and Pericarp Quantity: ' + str(round(pearsonr(combined_data.Norm_Prop_Pericarp_Retained, combined_data.Quantity)[0], 3)))
print('R of Pericarp Retained per Normalized Initial Content and Pericarp Quantity: ' + str(round(pearsonr(combined_data.Prop_Pericarp_Retained_Norm_Initial, combined_data.Quantity)[0], 3)))

# Plot out the best correlation and color by group
group_ref = pd.read_excel('../Data/Data_Grouping_XRef.xlsx', sheet_name = 'Sheet1')

combined_data = combined_data.merge(group_ref,
                                    on = 'Sample_ID',
                                    how = 'left')

# Determine colors for genetic groups of data
colors = {'Era_Inbred' : 'blue',
          'Era_F1' : 'lightblue',
          'Com_Hybrid' : 'red',
          'Popcorn_Inbred' : 'green',
          'Popcorn_Hybrid' : 'lightgreen',
          'W153R_Relative' : 'black'}

# Scale the data to have values between 0 and 1 for plot scale purposes
scaled_data = combined_data.copy()

for column in scaled_data.columns:
    if scaled_data[column].dtype.kind in 'biufc':
        scaled_data[column] = scaled_data[column] / scaled_data[column].max()

# Plot all of the correlations
plt.subplot(5,3,1)
plt.scatter(scaled_data.Coverage, scaled_data.Average_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.ylabel('Base')
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,4)
plt.scatter(scaled_data.Coverage, scaled_data.Normalized_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.ylabel('Norm')
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,7)
plt.scatter(scaled_data.Coverage, scaled_data.Proportion_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.ylabel('Prop')
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,10)
plt.scatter(scaled_data.Coverage, scaled_data.Norm_Prop_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.ylabel('Norm\nProp')
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,13)
plt.scatter(scaled_data.Coverage, scaled_data.Prop_Pericarp_Retained_Norm_Initial, c = scaled_data['Group'].map(colors), s = 0.5)
plt.ylabel('Prop\nNorm\nInitial')
plt.xlabel('Coverage')
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])

plt.subplot(5,3,2)
plt.scatter(scaled_data.Intensity, scaled_data.Average_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,5)
plt.scatter(scaled_data.Intensity, scaled_data.Normalized_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,8)
plt.scatter(scaled_data.Intensity, scaled_data.Proportion_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,11)
plt.scatter(scaled_data.Intensity, scaled_data.Norm_Prop_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,14)
plt.scatter(scaled_data.Intensity, scaled_data.Prop_Pericarp_Retained_Norm_Initial, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlabel('Intensity')
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])

plt.subplot(5,3,3)
plt.scatter(scaled_data.Quantity, scaled_data.Average_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,6)
plt.scatter(scaled_data.Quantity, scaled_data.Normalized_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,9)
plt.scatter(scaled_data.Quantity, scaled_data.Proportion_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,12)
plt.scatter(scaled_data.Quantity, scaled_data.Norm_Prop_Pericarp_Retained, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.subplot(5,3,15)
plt.scatter(scaled_data.Quantity, scaled_data.Prop_Pericarp_Retained_Norm_Initial, c = scaled_data['Group'].map(colors), s = 0.5)
plt.xlabel('Quantity')
plt.xlim((0,1.1))
plt.ylim((0,1.1))
plt.xticks([])
plt.yticks([])
plt.show()



model = np.polyfit(combined_data.Coverage, combined_data.Proportion_Pericarp_Retained, 1)
p = np.poly1d(model)
plt.scatter(combined_data.Coverage, combined_data.Proportion_Pericarp_Retained, c = 'black')
plt.plot(combined_data.Coverage, p(combined_data.Coverage), c = 'black')
plt.text(x = 0.83, y = 0.15, s = 'R = ' + str(round(pearsonr(combined_data.Coverage,
                                                          combined_data.Proportion_Pericarp_Retained)[0],
                                                 3)))
plt.text(x = 0.83, y = 0.09, s = 'P = ' + str(round(pearsonr(combined_data.Coverage,
                                                          combined_data.Proportion_Pericarp_Retained)[1],
                                                 3)))
plt.xlabel('')
plt.ylabel('')
plt.title('Annotated Pericarp Coverage')
plt.ylim(0,1)
plt.yticks([])
plt.show()


model = np.polyfit(combined_data.Quantity, combined_data.Proportion_Pericarp_Retained, 1)
p = np.poly1d(model)
plt.scatter(combined_data.Quantity, combined_data.Proportion_Pericarp_Retained, c = 'black')
plt.plot(combined_data.Quantity, p(combined_data.Quantity), c = 'black')
plt.text(x = 150, y = 0.15, s = 'R = ' + str(round(pearsonr(combined_data.Quantity,
                                                          combined_data.Proportion_Pericarp_Retained)[0],
                                                 3)))
plt.text(x = 150, y = 0.1, s = 'P = ' + str(round(pearsonr(combined_data.Quantity,
                                                          combined_data.Proportion_Pericarp_Retained)[1],
                                                 3)))
plt.xlabel('')
plt.ylabel('')
plt.title('Pericarp Proxy (Coverage x Blue)')
plt.yticks([])
plt.ylim(0,1)
plt.show()

