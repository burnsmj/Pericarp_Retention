#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:01:43 2022

@author: michael
"""
### Unsupervised Annotation Proof of Concept
### Michael Burns
### 7/21/22

# Purpose:  The purpose of this script is to process a few images to see if 
#           unsupervised clustering analyses can annotate images fairly
#           accurately.  I got this idea while looking for new image annotation
#           tools to replace CVAT.  I found a company called V7 that boasts
#           pixel-perfect annotations at 'unsupervised speed' which gave me the
#           idea to try clustering and then thresholding the clusters to sort
#           of predict which cluster is pericarp.  Background should be easy to
#           pick out, but pericarp from aleurone (especially with both white
#           and yellow) will be tougher.  My hope is that I can use a
#           clustering algorithm to cluster the image into three parts
#           (background, pericarp, aleurone).  If this looks fairly accurate,
#           I should then be able to threshold based on how much blue or green
#           is present.  If I can do that, then I can quantify how many pixels
#           are pericarp.  This doesn't need to be perfect as I have multiple
#           kernels per sample to try to average out errors.  

#           This will be done on my ground-truth dataset of images.  If the
#           annotation looks decent, I can try correlating pericarp retained to
#           pericarp coverage.

# Expected Workflow:    1. Import packages (numpy, skimage, sklearn, etc)
#                       2. Load image
#                       3. Process image through clustering algorithm
#                       4. Look at distribution of pixel values for each cluster (look at multiple color spaces)
#                       5. Find a threshold
#                       6. Repeat 2-5 for another few images to see if it is working
#                       7. Run 2-5 for all images and collect the % pericarp values
#                       8. Load dataset of ground truth values
#                       9. Correlate % pericarp coverage and pericarp retained
#                       10. Rejoice?

# Possible Next Steps:  The code seems to be working, and does okay in certain
#                       situations (xyz color space, ward linkage, CHI metric).
#                       This gets us to a correlation of about 0.551 between
#                       Normalized pericarp retained and assumed pericarp
#                       coverage.  This does not perform well on the images,
#                       but performs the best for correlations.  This is likely
#                       due to it restricting the number of 100% coverage
#                       samples that are processed.  Even considering stain
#                       blueness as a proxy for pericarp depth, the correlation
#                       gets worse. There may be an ideal combination of color
#                       space, linkage, and metric that I haven't found, but
#                       for now I think there are better uses of my time.  This
#                       would need to be run on MSI which means that I would
#                       need to upload all of the photos, change the file
#                       paths, and create a loop script that would inform this
#                       script what values to use for color space, linkage, and
#                       metric (7, 4, 3, respectively, 84 total iterations).

# Packages
import sys
import numpy as np
import pandas as pd
from skimage import io
from skimage import color
from skimage import transform
from sklearn import cluster
from sklearn import metrics
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Package Settings
plt.rcParams['figure.dpi'] = 300

# Command line arguments
cli_args = sys.argv
print(cli_args)
print(cli_args[1])
print(cli_args[2])
print(cli_args[3])

# Functions
def color_space_transformation(rgb_image, color_space = 'RGB'):
    if color_space == 'RGB':
        return rgb_image
    elif color_space == 'LAB':
        return color.rgb2lab(rgb_image)
    elif color_space == 'HSV':
        return color.rgb2hsv(rgb_image)
    elif color_space == 'XYZ':
        return color.rgb2xyz(rgb_image)
    elif color_space == 'LUV':
        return color.rgb2luv(rgb_image)
    elif color_space == 'YIQ':
        return color.rgb2yiq(rgb_image)
    elif color_space == 'YUV':
        return color.rgb2yuv(rgb_image)
    else:
        print('Color space provided was not recognized')

def downsample_image(image, factor):
    # Downsample the image 10x
    img_red_dwnsmpld = transform.rescale(image[:,:,0], factor, anti_aliasing = False)
    img_grn_dwnsmpld = transform.rescale(image[:,:,1], factor, anti_aliasing = False)
    img_blu_dwnsmpld = transform.rescale(image[:,:,2], factor, anti_aliasing = False)

    # Restack the channels
    img_dwnsmpld = np.dstack((img_red_dwnsmpld, img_grn_dwnsmpld, img_blu_dwnsmpld))

    return img_dwnsmpld

def cluster_pixels(image_ravel, show_metrics = False, metric = 'CHI', linkage = 'ward'):
    # Create cluster models with both 2 and 3 clusters
    cluster_model_k2 = cluster.AgglomerativeClustering(n_clusters = 2, linkage = linkage)
    cluster_model_k3 = cluster.AgglomerativeClustering(n_clusters = 3, linkage = linkage)

    # Fit cluster models above with the ravelled image pixels
    cluster_model_k2.fit(image_ravel)
    cluster_model_k3.fit(image_ravel)

    if metric == 'CHI':
        # Save model performance metrics to test later
        metric_k2 = metrics.calinski_harabasz_score(image_ravel, cluster_model_k2.labels_)
        metric_k3 = metrics.calinski_harabasz_score(image_ravel, cluster_model_k3.labels_)

        # Save image label mask based on which cluster number performed better
        if metric_k2 > metric_k3:
            image_labels = cluster_model_k2.labels_
            n_clust = 2
        elif metric_k3 > metric_k2:
            image_labels = cluster_model_k3.labels_
            n_clust = 3
        else:
            print('CH index values were the same, skipping this one for manual checking.')

        if show_metrics is True:
            print('Best Number of Clusters: ' + str(n_clust))
            print('K2 Index Score: ' + str(metric_k2))
            print('K3 Index Score: ' + str(metric_k3))

        return image_labels
    elif metric == 'SC':
        # Save model performance metrics to test later
        metric_k2 = metrics.silhouette_score(image_ravel, cluster_model_k2.labels_)
        metric_k3 = metrics.silhouette_score(image_ravel, cluster_model_k3.labels_)

        # Save image label mask based on which cluster number performed better
        if metric_k2 > metric_k3:
            image_labels = cluster_model_k2.labels_
            n_clust = 2
        elif metric_k3 > metric_k2:
            image_labels = cluster_model_k3.labels_
            n_clust = 3
        else:
            print('SC index values were the same, skipping this one for manual checking.')

        if show_metrics is True:
            print('Best Number of Clusters: ' + str(n_clust))
            print('K2 Index Score: ' + str(metric_k2))
            print('K3 Index Score: ' + str(metric_k3))

        return image_labels
    elif metric == 'DBI':
        # Save model performance metrics to test later
        metric_k2 = metrics.davies_bouldin_score(image_ravel, cluster_model_k2.labels_)
        metric_k3 = metrics.davies_bouldin_score(image_ravel, cluster_model_k3.labels_)

        # Save image label mask based on which cluster number performed better
        if metric_k2 < metric_k3:
            image_labels = cluster_model_k2.labels_
            n_clust = 2
        elif metric_k3 < metric_k2:
            image_labels = cluster_model_k3.labels_
            n_clust = 3
        else:
            print('SC index values were the same, skipping this one for manual checking.')

        if show_metrics is True:
            print('Best Number of Clusters: ' + str(n_clust))
            print('K2 Index Score: ' + str(metric_k2))
            print('K3 Index Score: ' + str(metric_k3))

        return image_labels
    else:
        print('A recognized metric was not given.')

def indicate_mask(image_ravel, labels, image_shape):
    pericarp, _, _ = determine_groups(image_ravel, labels)

    image_labeled = labels.reshape(image_shape[0:2])

    # Extract just the part that is theoretically pericarp
    image_indicate = np.zeros(image_shape[0:2])

    image_indicate[image_labeled == pericarp] = 1

    return image_indicate

def determine_groups(image_rgb_ravel, labels):
    groups = list(set(labels))
    red_content = []

    for group in groups:
        data = image_rgb_ravel[labels == group]
        red_content.append(data[:,0].mean())

    if max(groups) == 1:
        background_group = groups[np.argmin(red_content)]
        pericarp_group = 1 - background_group
        aleurone_group = 2

    elif max(groups) == 2:
        aleurone_group = groups[np.argmax(red_content)]
        background_group = groups[np.argmin(red_content)]
        assert aleurone_group != background_group
        pericarp_group = 3 - (aleurone_group + background_group)
    else:
        print('Incorrect number of groups')

    #print(pericarp_group)
    #print(aleurone_group)
    #print(background_group)

    return pericarp_group, aleurone_group, background_group

def percent_coverage(image_rgb_ravel, labels):
    if labels.max() == 1:
        prcnt_cvrg = 100
    elif labels.max() == 2:
        pericarp, aleurone, _ = determine_groups(image_rgb_ravel, labels)

        # Calculate percent coverage for 3 clusters
        prcnt_cvrg = (sum(labels == pericarp) / (sum(labels == pericarp) + sum(labels == aleurone))) * 100

    else:
        print('Image did not have either two or three clusters, ncluster = ' + str(np.argmax(labels)))

    return prcnt_cvrg

def pericarp_thickness(image_lab_ravel, image_rgb_ravel, labels):
    groups = list(set(labels))
    blu_int = []

    for group in groups:
        data = 255-image_lab_ravel[labels == group]
        blu_int.append(data[:,2].mean())

    pericarp, _, _ = determine_groups(image_rgb_ravel, labels)

    blu_int = np.array(blu_int)

    peri_thick = blu_int[groups == pericarp]
    #print(peri_thick[0])
    return peri_thick[0]

# Debugging - copy and paste into console
#image_lab_ravel = lowres_lab_raveled
#image_rgb_ravel = lowres_rgb_raveled
#labels = lowres_labels

# Glob the images
images = sorted(glob('/home/hirschc1/burns756/Pericarp/Data/Images/Annotation_Images/Ground_Truth_Cooked/Split_GT_Images/Ground_Truth_Images_*/*.tif'))

# Dictionary for storing coverage values in (genotype : coverages)
pericarp_data = {}

# Read in image
iter = 0
for img in images:
    if iter % 100 == 0:
        print('Working on Images ' + str(iter) + ' - ' + str(iter + 99) + ' out of ' + str(len(images)))

    image_rgb = io.imread(img, plugin = 'pil')
    image_lab = color.rgb2lab(image_rgb)
    image = color_space_transformation(image_rgb, str(cli_args[1]))

    # Downsample the image
    lowres_rgb = downsample_image(image_rgb, 0.1)
    lowres_lab = downsample_image(image_lab, 0.1)
    lowres_img = downsample_image(image, 0.1)
    #lowres_img = color.rgb2lab(lowres_rgb)

    # Ravel image to get each pixel as its own observation
    lowres_rgb_raveled = np.column_stack((lowres_rgb[:,:,0].ravel(), lowres_rgb[:,:,1].ravel(), lowres_rgb[:,:,2].ravel()))
    lowres_lab_raveled = np.column_stack((lowres_lab[:,:,0].ravel(), lowres_lab[:,:,1].ravel(), lowres_lab[:,:,2].ravel()))
    lowres_img_raveled = np.column_stack((lowres_img[:,:,0].ravel(), lowres_img[:,:,1].ravel(), lowres_img[:,:,2].ravel()))

    # Create and analyze cluster models
    lowres_labels = cluster_pixels(lowres_img_raveled, metric = str(cli_args[2]), linkage = str(cli_args[3]))

    # Recreate original image shape with labels
    image_labeled = lowres_labels.reshape(lowres_img.shape[0:2])

    pericarp_coverage = percent_coverage(lowres_rgb_raveled, lowres_labels)

    pericarp_thick_proxy = pericarp_thickness(lowres_lab_raveled, lowres_rgb_raveled, lowres_labels)

    pericarp_quantity = pericarp_thick_proxy * pericarp_coverage

    # Display the image progression of some samples
    #if iter % 100 == 0:
    #    plt.subplot(1,4,1)
    #    plt.imshow(image_rgb)
    #    plt.axis('off')
    #    plt.subplot(1,4,2)
    #    plt.imshow(lowres_rgb)
    #    plt.axis('off')
    #    plt.subplot(1,4,3)
    #    plt.imshow(image_labeled)
    #    plt.axis('off')
    #    plt.subplot(1,4,4)
    #    plt.imshow(indicate_mask(lowres_rgb_raveled, lowres_labels, lowres_img.shape), cmap = 'gray')
    #    plt.axis('off')
    #    plt.suptitle('Percent Pericarp Coverage: ' + str(round(pericarp_coverage, 2)), y = 0.75)
    #    plt.show()

    # Extract genotype from filename
    geno = img.split('/')[-1].split('_')[0]

    # Store the percent coverage in the coverages dictionary
    if geno not in pericarp_data:
        pericarp_data[geno] = {'coverage' : [pericarp_coverage],
                               'quantity' : [pericarp_quantity]}
    elif geno in pericarp_data:
        pericarp_data[geno]['coverage'].append(pericarp_coverage)
        pericarp_data[geno]['quantity'].append(pericarp_quantity)
    else:
        print('Could not add genotype or coverage to the dictionary')

    # Increase iter value
    iter += 1

# Determining average pericarp retention for each genotype
avg_coverages = {}
for geno in pericarp_data:
    avg_coverages[geno] = sum(pericarp_data[geno]['coverage']) / len(pericarp_data[geno]['coverage'])

avg_quantities = {}
for geno in pericarp_data:
    avg_quantities[geno] = sum(pericarp_data[geno]['quantity']) / len(pericarp_data[geno]['quantity'])

# Turn average pericarp retention dictionary into a dataframe
percent_coverage_df = pd.DataFrame(avg_coverages.items(), columns = ['Sample_ID', 'Percent_Coverage'])

quantity_df = pd.DataFrame(avg_quantities.items(), columns = ['Sample_ID', 'Pericarp_Quantity'])

# Read in ground truth data and kernel mass data
cooked_data = pd.read_excel('/home/hirschc1/burns756/Pericarp/Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Rapid_Cooked')
kernel_masses = pd.read_excel('/home/hirschc1/burns756/Pericarp/Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Kernel_Mass')
imbibed_data = pd.read_excel('/home/hirschc1/burns756/Pericarp/Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Imbibed')

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
                                             how = 'left').merge(percent_coverage_df,
                                                                 on = 'Sample_ID',
                                                                 how = 'left'). merge(quantity_df,
                                                                                      on = 'Sample_ID',
                                                                                      how = 'left')

# Determine the normalized proportion of pericarp retained
combined_data['Normalized_Pericarp_Retained'] = combined_data.Average_Pericarp_Retained / combined_data.Average_Kernel_Mass
combined_data['Proportion_Pericarp_Retained'] = combined_data.Average_Pericarp_Retained / combined_data.Average_Pericarp_Initial
combined_data['Norm_Prop_Pericarp_Retained'] = combined_data.Proportion_Pericarp_Retained / combined_data.Average_Kernel_Mass

# Determine correlations between percent coverage, average pericarp retained, and normalized pericarp retained
print('R of Pericarp Retained and Assumed Pericarp Coverage: ' + str(round(pearsonr(combined_data.Average_Pericarp_Retained, combined_data.Percent_Coverage)[0], 3)))
print('R of Normalized Pericarp Retained and Assumed Pericarp Coverage: ' + str(round(pearsonr(combined_data.Normalized_Pericarp_Retained, combined_data.Percent_Coverage)[0], 3)))
print('R of Proportion Pericarp Retained and Assumed Pericarp Coverage: ' + str(round(pearsonr(combined_data.Proportion_Pericarp_Retained, combined_data.Percent_Coverage)[0], 3)))
print('R of Normalized Pericarp Retained and Assumed Pericarp Coverage: ' + str(round(pearsonr(combined_data.Norm_Prop_Pericarp_Retained, combined_data.Percent_Coverage)[0], 3)))

print('R of Pericarp Retained and Assumed Pericarp Quantity: ' + str(round(pearsonr(combined_data.Average_Pericarp_Retained, combined_data.Pericarp_Quantity)[0], 3)))
print('R of Normalized Pericarp Retained and Assumed Pericarp Quantity: ' + str(round(pearsonr(combined_data.Normalized_Pericarp_Retained, combined_data.Pericarp_Quantity)[0], 3)))
print('R of Proportion Pericarp Retained and Assumed Pericarp Quantity: ' + str(round(pearsonr(combined_data.Proportion_Pericarp_Retained, combined_data.Pericarp_Quantity)[0], 3)))
print('R of Normalized Pericarp Retained and Assumed Pericarp Quantity: ' + str(round(pearsonr(combined_data.Norm_Prop_Pericarp_Retained, combined_data.Pericarp_Quantity)[0], 3)))

# Saving the average pericarp coverage per sample
#combined_data.to_csv('~/Desktop/Test_Coverages.csv', index = False)
