#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:28:10 2022

@author: michael
"""
### Annotation EDA
### Michael Burns
### 10/20/22

# Purpose:  To determine if there are any easy data differences between the two
#           groups of pericarp and non-pericarp.  This could be differences in
#           R,G,B,H,S,V,L,A,B,etc, or a difference in some other metric that
#           may come up.  A lot of this exploration will be based on tabular
#           data, so the first step will be to read in the non-flipped, 0 deg
#           rotation images and their masks.  Following this, I can extract the
#           channels as needed.  It may also be helpful to utilize a PCA plot
#           of the different channels to see if I can find separation anywhere.

# Packages
from time import time
import pandas as pd
import numpy as np
from random import sample, choices
from glob import glob
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, cohen_kappa_score

# Functions
def rgb_ravel(rgb_image, mask_image, bkgd_rm = True):
    img_rgb_r = rgb_image[:,:,0].ravel()#[subset_index]
    img_rgb_g = rgb_image[:,:,1].ravel()#[subset_index]
    img_rgb_b = rgb_image[:,:,2].ravel()#[subset_index]
    
    if bkgd_rm is True:
        mask_rav = mask_image.ravel()
        rgb_data = pd.DataFrame({'RGB_R' : img_rgb_r[mask_rav != 0].astype(np.uint8),
                                 'RGB_G' : img_rgb_g[mask_rav != 0].astype(np.uint8),
                                 'RGB_B' : img_rgb_b[mask_rav != 0].astype(np.uint8)})
    else:
        rgb_data = pd.DataFrame({'RGB_R' : img_rgb_r.astype(np.uint8),
                                 'RGB_G' : img_rgb_g.astype(np.uint8),
                                 'RGB_B' : img_rgb_b.astype(np.uint8)})
    
    return rgb_data

def lab_ravel(rgb_image, mask_image, bkgd_rm = True):
    img_lab = color.rgb2lab(rgb_image)
    img_lab_l = img_lab[:,:,0].ravel()#[subset_index]
    img_lab_a = img_lab[:,:,1].ravel()#[subset_index]
    img_lab_b = img_lab[:,:,2].ravel()#[subset_index]
    
    if bkgd_rm is True:
        mask_rav = mask_image.ravel()
        lab_data = pd.DataFrame({'LAB_L' : img_lab_l[mask_rav != 0].astype(np.float32),
                                 'LAB_A' : img_lab_a[mask_rav != 0].astype(np.float32),
                                 'LAB_B' : img_lab_b[mask_rav != 0].astype(np.float32)})
    else:
        lab_data = pd.DataFrame({'LAB_L' : img_lab_l.astype(np.float32),
                                 'LAB_A' : img_lab_a.astype(np.float32),
                                 'LAB_B' : img_lab_b.astype(np.float32)})
    
    return lab_data

def hsv_ravel(rgb_image, mask_image, bkgd_rm = True):
    img_hsv = color.rgb2hsv(rgb_image)
    img_hsv_h = img_hsv[:,:,0].ravel()#[subset_index]
    img_hsv_s = img_hsv[:,:,1].ravel()#[subset_index]
    img_hsv_v = img_hsv[:,:,2].ravel()#[subset_index]
    
    if bkgd_rm is True:
        mask_rav = mask_image.ravel()
        hsv_data = pd.DataFrame({'HSV_H' : img_hsv_h[mask_rav != 0].astype(np.float32),
                                 'HSV_S' : img_hsv_s[mask_rav != 0].astype(np.float32),
                                 'HSV_V' : img_hsv_v[mask_rav != 0].astype(np.float32)})
    else:
        hsv_data = pd.DataFrame({'HSV_H' : img_hsv_h.astype(np.float32),
                                 'HSV_S' : img_hsv_s.astype(np.float32),
                                 'HSV_V' : img_hsv_v.astype(np.float32)})
    
    return hsv_data

def xyz_ravel(rgb_image, mask_image, bkgd_rm = True):
    img_xyz = color.rgb2xyz(rgb_image)
    img_xyz_x = img_xyz[:,:,0].ravel()#[subset_index]
    img_xyz_y = img_xyz[:,:,1].ravel()#[subset_index]
    img_xyz_z = img_xyz[:,:,2].ravel()#[subset_index]
    
    if bkgd_rm is True:
        mask_rav = mask_image.ravel()
        xyz_data = pd.DataFrame({'XYZ_X' : img_xyz_x[mask_rav != 0].astype(np.float32),
                                 'XYZ_Y' : img_xyz_y[mask_rav != 0].astype(np.float32),
                                 'XYZ_Z' : img_xyz_z[mask_rav != 0].astype(np.float32)})
    else:
        xyz_data = pd.DataFrame({'XYZ_X' : img_xyz_x.astype(np.float32),
                                 'XYZ_Y' : img_xyz_y.astype(np.float32),
                                 'XYZ_Z' : img_xyz_z.astype(np.float32)})
    
    return xyz_data

def luv_ravel(rgb_image, mask_image, bkgd_rm = True):
    img_luv = color.rgb2luv(rgb_image)
    img_luv_l = img_luv[:,:,0].ravel()#[subset_index]
    img_luv_u = img_luv[:,:,1].ravel()#[subset_index]
    img_luv_v = img_luv[:,:,2].ravel()#[subset_index]
    
    if bkgd_rm is True:
        mask_rav = mask_image.ravel()
        luv_data = pd.DataFrame({'LUV_L' : img_luv_l[mask_rav != 0].astype(np.float32),
                                 'LUV_U' : img_luv_u[mask_rav != 0].astype(np.float32),
                                 'LUV_V' : img_luv_v[mask_rav != 0].astype(np.float32)})
    else:
        luv_data = pd.DataFrame({'LUV_L' : img_luv_l.astype(np.float32),
                                 'LUV_U' : img_luv_u.astype(np.float32),
                                 'LUV_V' : img_luv_v.astype(np.float32)})
    
    return luv_data

def yuv_ravel(rgb_image, mask_image, bkgd_rm = True):
    img_yuv = color.rgb2yuv(rgb_image)
    img_yuv_y = img_yuv[:,:,0].ravel()#[subset_index]
    img_yuv_u = img_yuv[:,:,1].ravel()#[subset_index]
    img_yuv_v = img_yuv[:,:,2].ravel()#[subset_index]
    
    if bkgd_rm is True:
        mask_rav = mask_image.ravel()
        yuv_data = pd.DataFrame({'YUV_Y' : img_yuv_y[mask_rav != 0].astype(np.float32),
                                 'YUV_U' : img_yuv_u[mask_rav != 0].astype(np.float32),
                                 'YUV_V' : img_yuv_v[mask_rav != 0].astype(np.float32)})
    else:
        yuv_data = pd.DataFrame({'YUV_Y' : img_yuv_y.astype(np.float32),
                                 'YUV_U' : img_yuv_u.astype(np.float32),
                                 'YUV_V' : img_yuv_v.astype(np.float32)})
    return yuv_data
    
def expand_and_ravel(rgb_image, mask_image, bkgd_rm = True, RGB = True, LAB = True,
                     HSV = True, XYZ = True, LUV = True, YUV = True):
    # May need to add a subset index part to the function later, so I kept the code for that commented out.
    color_data = pd.DataFrame()
    if RGB is True:
        color_data = pd.concat([color_data, rgb_ravel(rgb_image, mask_image)], axis = 1)
    if LAB is True:
        color_data = pd.concat([color_data, lab_ravel(rgb_image, mask_image)], axis = 1)
    if HSV is True:
        color_data = pd.concat([color_data, hsv_ravel(rgb_image, mask_image)], axis = 1)
    if XYZ is True:
        color_data = pd.concat([color_data, xyz_ravel(rgb_image, mask_image)], axis = 1)
    if LUV is True:
        color_data = pd.concat([color_data, luv_ravel(rgb_image, mask_image)], axis = 1)
    if YUV is True:
        color_data = pd.concat([color_data, yuv_ravel(rgb_image, mask_image)], axis = 1)
    
    return color_data

def density_intersect(data_variable, data_label, plot = True):
    # Create density distrubtions
    kernel_1 = gaussian_kde(data_variable[data_label == 1])
    kernel_2 = gaussian_kde(data_variable[data_label == 2])
    
    # Create the space the density will be applied to
    x = np.linspace(start = min(data_variable), stop = max(data_variable), num = 1000)
    
    # Determine the densities
    y_1 = kernel_1(x)
    y_2 = kernel_2(x)

    # Take the first derivative of the difference in y values    
    deriv_1 = np.diff(abs(y_1 - y_2)) / np.diff(x)
    
    # Take the second derivatve of the difference in values from the first derivative
    derive_2 = np.diff(deriv_1) / np.diff(x[1:])
    
    # Determine the intersect point as the largest value of the second derivative
    intersect = x[:-2][derive_2 == max(derive_2)][0]
    
    if plot is True:
        # Plot out the distributions
        plt.plot(x, y_1, color = 'brown')
        plt.plot(x, y_2, color = 'blue')
        plt.vlines(intersect, 0, max(max(y_1), max(y_2)), colors = 'black')
        plt.title(var)
        plt.xlabel('Channel Value')
        plt.ylabel('Density')
        plt.show()
    
    return intersect
        
    

# Read in data
images = sorted(glob('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Images/*.png'))
# SPLIT BY GERM DIRECTION AND KERNEL COLOR
germ_dir = pd.read_excel('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Germ_Directions.xlsx') # this needs to be remade to include image file name

# Loop through images
allow_loop = True # This loop doesn't need to run every time since I save the data.  Change this to True if the data needs to be collected again.
if allow_loop is True:
    # GO BACK TO SAMPLING IMAGES
    iter = 0
    start_time = time()
    last_time = time()
    times = []
    for image in images:
        # Progress Tracker
        if iter % 100 == 0:
            if iter == 0:
                print('\rWorking on iterations ' + str(iter) + ' - ' + str(min(iter + 99, len(images))) + ' of ' + str(len(images)), end = '\r')
            else:
                times.append(time() - last_time)
                print('\rWorking on iterations ' + str(iter) + ' - ' + str(min(iter + 99, len(images))) + ' of ' + str(len(images)), '|',
                      'Time for last loop: ' + str(round(times[-1])) + 'seconds', '|',
                      'Approximate time remaining: ' + str(round(((len(images) - iter) * (np.mean(times) / 100)) / 60, 2)) + 'minutes',
                      end = '\r')
                last_time = time()
        iter += 1
        
        #################
        # Read in image #
        #################
        img_rgb = io.imread(image)
        
        #########################
        # Extract the file name #
        #########################
        fname = image.split('/')[-1]
        
        ##########################
        # Read in the mask image #
        ##########################
        mask = io.imread('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Masks/' + fname, as_gray = True)
        
        ##############################
        # Ravel and correct the mask #
        ##############################
        mask = mask / 127 # Needed to convert values back to 0 1 and 2
        mask = mask.astype(np.uint8) # Needed to round the values for 2
    
        ##########################
        # Set background removal #
        ##########################
        bkgd_rm = True # Need to set it here so I can use it for both the functions and filtering the label data
    
        data = expand_and_ravel(img_rgb, mask, bkgd_rm = bkgd_rm) # Ravel multiple color spaces together
        
        ################################
        # Add image data to data frame #
        ################################
        data['fname'] = fname
        if bkgd_rm is True:
            data['label'] = mask.ravel()[mask.ravel() != 0]
        else:
            data['label'] = mask.ravel().astype(np.uint8)
        data['shape_0'] = np.array(img_rgb.shape[0], dtype = np.uint8)
        data['shape_1'] = np.array(img_rgb.shape[1], dtype = np.uint8)
        
        ##############################
        # Write out data to csv file #
        ##############################
        if iter == 1:
            data.to_csv('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Pixel_Info.csv', mode = 'w', index = False)
        else:
            data.to_csv('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Pixel_Info.csv', mode = 'a', index = False, header = False)


# Read in Pixel Data
data = pd.read_csv('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Pixel_Info.csv',
                   index_col = False,
                   dtype = {'RGB_R' : np.uint8,
                            'RGB_G' : np.uint8,
                            'RGB_B' : np.uint8,
                            'LAB_L' : np.float32,
                            'LAB_A' : np.float32,
                            'LAB_B' : np.float32,
                            'HSV_H' : np.float32,
                            'HSV_S' : np.float32,
                            'HSV_V' : np.float32,
                            'XYZ_X' : np.float32,
                            'XYZ_Y' : np.float32,
                            'XYZ_Z' : np.float32,
                            'LUV_L' : np.float32,
                            'LUV_U' : np.float32,
                            'LUV_V' : np.float32,
                            'YUV_Y' : np.float32,
                            'YUV_U' : np.float32,
                            'YUV_V' : np.float32,
                            'fname' : object, 
                            'label' : np.uint8,
                            'shape_0' : np.uint8,
                            'shape_1' : np.uint8})

print(data.shape)

# Extract the column names and only keep the channel ones
col_names = data.columns
num_col_names = col_names[:-4]

##########################
### Threshold by Color ###
##########################

# Create empty lists to fill
channel = []
intersections = []
iteration = []
n_loops = 100

for i in range(n_loops):
    # Progress Tracker
    if i % 10 == 0:
        print('\rWorking on iterations ' + str(i) + ' - ' + str(min(i + 9, n_loops)) + ' of ' + str(n_loops), end = '\r')

    # Subsample data
    data_small = data.sample(10000)
    
    for var in num_col_names:
        # Calculate the intersection point
        intersect = density_intersect(data_small[var], data_small.label, plot = False)
        
        # Save the data from each iteration
        channel.append(var)
        intersections.append(intersect)
        iteration.append(i)

int_data = pd.DataFrame({'Channel' : channel,
                         'Intersection' : intersections,
                         'Iteration' : iteration})

avg_ints = int_data.groupby('Channel').mean().Intersection

# We now have a way to find intersections for ideal data splitting

# Correlations between channels
data_small = data.sample(10000)

data_small_chan = data_small[num_col_names]

data_small_chan['label'] = data_small.label

cor = abs(data_small_chan.corr())
sns.heatmap(cor, cmap = plt.cm.Reds)
plt.show()

target_cors = cor[cor.label > 0.5]

# Based on the code above, it appears that label is most highly correlated with
# just 9 of the color spaces chosen (abs(R) > 0.5).  These are all highly
# correlated with each other, so we should either choose one to threshold with
# (XYZ_X has the highest correlation), or use PCA or PLSDA to account for
# multicolinearity in the dataset.
data['xpred'] = 1
data.xpred[data.XYZ_X <= avg_ints[avg_ints.index == 'XYZ_X'][0]] = 2

print('XYZ X based prediction metrics:')
print('Accuracy: ', accuracy_score(data.label, data.xpred))
print('Kappa: ', cohen_kappa_score(data.label, data.xpred))
print('Balanced Accuracy: ', balanced_accuracy_score(data.label, data.xpred))
print('ROC AUC: ', roc_auc_score(data.label, data.xpred))
print('F1 Score: ', f1_score(data.label, data.xpred))
print('Confusion Matrix:\n', confusion_matrix(data.label, data.xpred))

# Lets try creating a PCA of the color space data to see if we can pick up on
# any of these differences.

#############################
### Threshold on PCA Data ###
#############################

# PCA
pca = PCA(n_components = 18)

# Extract the PC data
PCs = pd.DataFrame(data = pca.fit_transform(X = data[num_col_names]),
             columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9',
                        'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18'])

# PCA Loadings
loadings = pd.DataFrame(data = pca.components_ * np.sqrt(pca.explained_variance_),
             columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
                        'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
                        'PC16', 'PC17', 'PC18'])
loadings['color'] = ['R', 'G', 'Bl', 'L','A', 'B', 'H', 'S', 'V',
                     'X', 'Y', 'Z', 'Ll', 'U', 'Vv', 'Yy', 'Uu', 'Vvv']

# PVE by each PC
plt.plot(range(1,19), pca.explained_variance_ratio_)
plt.xlabel('Number of Principal Components')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# Variable for the number of significant PCs
n_sig_pcs = 3

# Add a labels column
PCs['label'] = list(data.label)

# Plot the first 4 principal components colored by label
pcs_small_index = sample(range(PCs.shape[0]), 2000)
pcs_small = PCs.iloc[pcs_small_index]

iter = 1

for r in range(n_sig_pcs):
    for c in range(n_sig_pcs):
        if c <= r:
            print(pcs_small.columns[c])
            print(pcs_small.columns[r])
            plt.subplot(n_sig_pcs, n_sig_pcs, iter)
            plt.scatter(pcs_small.iloc[:,c], pcs_small.iloc[:,r],
                        c = pcs_small.label,
                        alpha = 0.4)
            if r == n_sig_pcs - 1:
                plt.xlabel(pcs_small.columns[c])
            if c == 0:
                plt.ylabel(pcs_small.columns[r])
            if c == 0 and r == 0:
                plt.vlines(x = 65, ymin = 0, ymax = 100)
            
        iter += 1

plt.tight_layout()
plt.show()

# Create empty lists to fill
pcs = []
pc_ints = []
pc_iters = []
n_loops = 100

for i in range(n_loops):
    # Progress Tracker
    if i % 10 == 0:
        print('\rWorking on iterations ' + str(i) + ' - ' + str(min(i + 9, n_loops)) + ' of ' + str(n_loops), end = '\r')

    # Subsample data
    pc_small = PCs.sample(10000)
    
    for var in pc_small.columns[:-1]:
        # Calculate the intersection point
        intersect = density_intersect(pc_small[var], pc_small.label, plot = False)
        
        # Save the data from each iteration
        pcs.append(var)
        pc_ints.append(intersect)
        pc_iters.append(i)

pc_int_data = pd.DataFrame({'PC' : pcs,
                         'Intersection' : pc_ints,
                         'Iteration' : pc_iters})

pc_avg_ints = pc_int_data.groupby('PC').mean().Intersection

# There appears to be some separation of pixel class for PC1 with PC2-PC4,
# primarily along the first PC axis.  The remaining PCs do not seem to be able
# to separate the pixel classes though, which may mean that the first PC is all
# that is really needed to separate the classes.

n1 = sum(PCs.label == 1)
n2 = sum(PCs.label == 2)

prop_1 = n1 / (n1+n2)
prop_2 = n2 / (n1+n2)

PCs['pred'] = 1
PCs['rand'] = choices([1,2], weights = (prop_1, prop_2), k = PCs.shape[0])
if n1 > n2:
    PCs['maj_pred'] = 1
else:
    PCs['maj_pred'] = 2
 
PCs.pred[PCs.PC1 <= pc_avg_ints[pc_avg_ints.index == 'PC1'][0]] = 2

print('PC1 based prediction metrics:')
print('Accuracy: ', accuracy_score(PCs.label, PCs.pred))
print('Kappa: ', cohen_kappa_score(PCs.label, PCs.pred))
print('Balanced Accuracy: ', balanced_accuracy_score(PCs.label, PCs.pred))
print('ROC AUC: ', roc_auc_score(PCs.label, PCs.pred))
print('F1 Score: ', f1_score(PCs.label, PCs.pred))
print('Confusion Matrix:\n', confusion_matrix(PCs.label, PCs.pred))

print('\nWeighted random sampling based prediction metrics:')
print('Accuracy: ', accuracy_score(PCs.label, PCs.rand))
print('Kappa: ', cohen_kappa_score(PCs.label, PCs.rand))
print('Balanced Accuracy: ', balanced_accuracy_score(PCs.label, PCs.rand))
print('ROC AUC: ', roc_auc_score(PCs.label, PCs.rand))
print('F1 Score: ', f1_score(PCs.label, PCs.rand))
print('Confusion Matrix:\n', confusion_matrix(PCs.label, PCs.rand))

print('\nMajor class based prediction metrics:')
print('Accuracy: ', accuracy_score(PCs.label, PCs.maj_pred))
print('Kappa: ', cohen_kappa_score(PCs.label, PCs.maj_pred))
print('Balanced Accuracy: ', balanced_accuracy_score(PCs.label, PCs.maj_pred))
print('ROC AUC: ', roc_auc_score(PCs.label, PCs.maj_pred))
print('F1 Score: ', f1_score(PCs.label, PCs.maj_pred))
print('Confusion Matrix:\n', confusion_matrix(PCs.label, PCs.maj_pred))

# I tried making equal class sizes, but I am not sure this is a worthwhile
# endevour since it is unrealistic.  I can focus on using better metrics and
# creating thresholds based on the real data.  Because of this I decided it was
# not worth keeping the code and extending the run time of this script.

############################
### Center and Scale PCA ###
############################
scaler = StandardScaler()
norm_data = pd.DataFrame(data = scaler.fit_transform(data[num_col_names]),
                         columns = num_col_names)

# PCA
norm_pca = PCA(n_components = 18)

# Extract the PC data
norm_pcs = pd.DataFrame(data = norm_pca.fit_transform(X = norm_data[num_col_names]),
             columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9',
                        'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18'])

# PCA Loadings
loadings = pd.DataFrame(data = norm_pca.components_ * np.sqrt(norm_pca.explained_variance_),
             columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
                        'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
                        'PC16', 'PC17', 'PC18'])
loadings['color'] = ['R', 'G', 'Bl', 'L','A', 'B', 'H', 'S', 'V',
                     'X', 'Y', 'Z', 'Ll', 'U', 'Vv', 'Yy', 'Uu', 'Vvv']

# PVE by each PC
plt.plot(range(1,19), norm_pca.explained_variance_ratio_)
plt.xlabel('Number of Principal Components')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# Variable for the number of significant PCs
n_sig_pcs = 3

# Add a labels column
norm_pcs['label'] = list(data.label)

# Plot the first 4 principal components colored by label
pcs_small_index = sample(range(norm_pcs.shape[0]), 2000)
pcs_small = norm_pcs.iloc[pcs_small_index]

iter = 1

for r in range(n_sig_pcs):
    for c in range(n_sig_pcs):
        if c <= r:
            print(pcs_small.columns[c])
            print(pcs_small.columns[r])
            plt.subplot(n_sig_pcs, n_sig_pcs, iter)
            plt.scatter(pcs_small.iloc[:,c], pcs_small.iloc[:,r],
                        c = pcs_small.label,
                        alpha = 0.4)
            if r == n_sig_pcs - 1:
                plt.xlabel(pcs_small.columns[c])
            if c == 0:
                plt.ylabel(pcs_small.columns[r])
            if c == 0 and r == 0:
                plt.vlines(x = 2.5, ymin = -5, ymax = 7)
            
        iter += 1

plt.tight_layout()
plt.show()

npcs = []
npc_ints = []
npc_iters = []
n_loops = 100

for i in range(n_loops):
    # Progress Tracker
    if i % 10 == 0:
        print('\rWorking on iterations ' + str(i) + ' - ' + str(min(i + 9, n_loops)) + ' of ' + str(n_loops), end = '\r')

    # Subsample data
    npc_small = norm_pcs.sample(10000)
    
    for var in npc_small.columns[:-1]:
        # Calculate the intersection point
        intersect = density_intersect(npc_small[var], npc_small.label, plot = False)
        
        # Save the data from each iteration
        npcs.append(var)
        npc_ints.append(intersect)
        npc_iters.append(i)

npc_int_data = pd.DataFrame({'PC' : npcs,
                         'Intersection' : npc_ints,
                         'Iteration' : npc_iters})

npc_avg_ints = npc_int_data.groupby('PC').mean().Intersection

n1 = sum(norm_pcs.label == 1)
n2 = sum(norm_pcs.label == 2)

prop_1 = n1 / (n1+n2)
prop_2 = n2 / (n1+n2)

norm_pcs['pred'] = 0
norm_pcs['rand'] = choices([1,2], weights = (prop_1, prop_2), k = norm_pcs.shape[0])
if n1 > n2:
    norm_pcs['maj_pred'] = 1
else:
    norm_pcs['maj_pred'] = 2
 
norm_pcs.pred[norm_pcs.PC1 <= 3] = 2
norm_pcs.pred[norm_pcs.PC1 > 3] = 1

print('Normalized PC1 based prediction metrics:')
print('Accuracy: ', accuracy_score(norm_pcs.label, norm_pcs.pred))
print('Kappa: ', cohen_kappa_score(norm_pcs.label, norm_pcs.pred))
print('Balanced Accuracy: ', balanced_accuracy_score(norm_pcs.label, norm_pcs.pred))
print('ROC AUC: ', roc_auc_score(norm_pcs.label, norm_pcs.pred))
print('F1 Score: ', f1_score(norm_pcs.label, norm_pcs.pred))
print('Confusion Matrix:\n', confusion_matrix(norm_pcs.label, norm_pcs.pred))


####################################
### PLSDA to Predict Pixel Class ###
####################################
# 2 Latent Variables, no scaling
plsr = PLSRegression(n_components=3, scale=True)

# PLS-DA algorithm
plsr.fit(data_small[num_col_names], data_small.label)

data['plsr_pred'] = np.round(plsr.predict(data[num_col_names]))

print('PLSR based prediction metrics:')
print('Accuracy: ', accuracy_score(data.label, data.plsr_pred))
print('Kappa: ', cohen_kappa_score(data.label, data.plsr_pred))
print('Balanced Accuracy: ', balanced_accuracy_score(data.label, data.plsr_pred))
print('ROC AUC: ', roc_auc_score(data.label, data.plsr_pred))
print('F1 Score: ', f1_score(data.label, data.plsr_pred))
print('Confusion Matrix:\n', confusion_matrix(data.label, data.plsr_pred))









iter = 0
for image in images:
    if iter % 100 == 0:
        print('Working on iterations ' + str(iter) + ' - ' + str(iter + 99) + ' of ' + str(len(images)))
    
        # Plot images to see how boundaries look
        img = io.imread(image)
        fname = image.split('/')[-1]
        mask = io.imread('/Users/michael/Desktop/Grad_School/Research/Projects/Pericarp/Data/Images/Low_Res_Images/Masks/' + fname, as_gray = True)
        mask_adj = mask / 127
        mask_adj = mask_adj.astype(int)
        
        img_rgb = img
        img_lab = color.rgb2lab(img_rgb)
        img_hsv = color.rgb2hsv(img_rgb)
        img_xyz = color.rgb2xyz(img_rgb)
        img_luv = color.rgb2luv(img_rgb)
        img_yuv = color.rgb2yuv(img_rgb)
        
        img_r_rav = img_rgb[:,:,0].ravel()#[subset_index]
        img_g_rav = img_rgb[:,:,1].ravel()#[subset_index]
        img_bl_rav = img_rgb[:,:,2].ravel()#[subset_index]
        img_l_rav = img_lab[:,:,0].ravel()#[subset_index]
        img_a_rav = img_lab[:,:,1].ravel()#[subset_index]
        img_b_rav = img_lab[:,:,2].ravel()#[subset_index]
        img_h_rav = img_hsv[:,:,0].ravel()#[subset_index]
        img_s_rav = img_hsv[:,:,1].ravel()#[subset_index]
        img_v_rav = img_hsv[:,:,2].ravel()#[subset_index]
        img_x_rav = img_xyz[:,:,0].ravel()#[subset_index]
        img_y_rav = img_xyz[:,:,1].ravel()#[subset_index]
        img_z_rav = img_xyz[:,:,2].ravel()#[subset_index]
        img_ll_rav = img_luv[:,:,0].ravel()#[subset_index]
        img_u_rav = img_luv[:,:,1].ravel()#[subset_index]
        img_vv_rav = img_luv[:,:,2].ravel()#[subset_index]
        img_yy_rav = img_yuv[:,:,0].ravel()#[subset_index]
        img_uu_rav = img_yuv[:,:,1].ravel()#[subset_index]
        img_vvv_rav = img_yuv[:,:,2].ravel()#[subset_index]
        
        data = pd.DataFrame({'R' : img_r_rav,
                             'G' : img_g_rav,
                             'Bl' : img_bl_rav,
                             'L' : img_l_rav,
                             'A' : img_a_rav,
                             'B' : img_b_rav,
                             'H' : img_h_rav,
                             'S' : img_s_rav,
                             'V' : img_v_rav,
                             'X' : img_x_rav,
                             'Y' : img_y_rav,
                             'Z' : img_z_rav,
                             'Ll' : img_ll_rav,
                             'U' : img_u_rav,
                             'Vv' : img_vv_rav,
                             'Yy' : img_yy_rav,
                             'Uu' : img_uu_rav,
                             'Vvv' : img_vvv_rav})
        
        norm_data = pd.DataFrame(data = scaler.fit_transform(data),
                                 columns = num_col_names)
        
        new_pc = pd.DataFrame(data = norm_pca.transform(norm_data),
                     columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9',
                                'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18'])
        
        
        new_pc['label'] = mask_adj.ravel()
        new_pc['pred'] = 0
        new_pc.pred[np.logical_and(new_pc.PC1 <= 3, new_pc.PC1 > -2)] = 2
        new_pc.pred[new_pc.PC1 > 3] = 1
        
        pc_labels = np.array(new_pc.pred)
        
        pc_mask = pc_labels.reshape(mask.shape)
        
        print('PC1 based prediction metrics:')
        print('Accuracy: ', accuracy_score(new_pc.label, new_pc.pred))
        print('Kappa: ', cohen_kappa_score(new_pc.label, new_pc.pred))
        #print('Balanced Accuracy: ', balanced_accuracy_score(new_pc.label, new_pc.pred))
        #print('ROC AUC: ', roc_auc_score(new_pc.label, new_pc.pred))
        #print('F1 Score: ', f1_score(new_pc.label, new_pc.pred))
        print('Confusion Matrix:\n', confusion_matrix(new_pc.label, new_pc.pred))
        
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(mask_adj, cmap = 'gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(pc_mask, cmap = 'gray')
        plt.axis('off')
        plt.show()
    iter += 1

# Next steps:
#   Check if there are any combinations of color spaces that make better PCs.
#   Separate images based on germ orientation
#   Plot images to see how the boundaries look
