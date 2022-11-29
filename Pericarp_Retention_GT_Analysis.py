#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:48:29 2022

@author: michael
"""
### Pericarp Ground Truth Analysis
### Michael Burns
### 6/22/22

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Table display setting change
pd.set_option('display.max_columns', None)

# Figure display settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (8,8)

# Read in data
cooked_data = pd.read_excel('../Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Rapid_Cooked')
imbibed_data = pd.read_excel('../Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Imbibed')
kernel_masses = pd.read_excel('../Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Kernel_Mass')
het_groups = pd.read_excel('../Data/Pericarp_Removal_Samples.xlsx', sheet_name = 'Heterotic_Groups')

# Filter for non-stained samples
cooked_data = cooked_data[cooked_data.Stained == 'N']

# Select columns needed
cooked_data = cooked_data[['Sample_ID', 'Pericarp_Remover', 'Average_Pericarp_Retained']]
imbibed_data = imbibed_data[['Sample_ID', 'Peeler_Name', 'Average_Pericarp_Initial']]
kernel_masses = kernel_masses[['Sample_ID', 'Weigher', 'Average_Kernel_Mass']]

# Combine datasets
data = cooked_data.merge(imbibed_data,
                         on = 'Sample_ID',
                         how = 'left').merge(kernel_masses,
                                             on = 'Sample_ID',
                                             how = 'left').merge(het_groups,
                                                                 on = 'Sample_ID',
                                                                 how = 'left')

# Calculate proportion retained
data['Prop_Retained'] = data.Average_Pericarp_Retained / data.Average_Pericarp_Initial
data['Initial_Pericarp_Prop'] = data.Average_Pericarp_Initial / data.Average_Kernel_Mass
data['Prop_Retained_Norm'] = data.Prop_Retained / data.Average_Kernel_Mass

# Plot distribution of proportion retained
plt.subplot(2,3,1)
plt.hist(data.Average_Pericarp_Initial)
plt.title('Initial Pericarp\n(10k Avg)')
plt.subplot(2,3,2)
plt.hist(data.Average_Pericarp_Retained)
plt.title('Pericarp Retained\n(10k Avg)')
plt.subplot(2,3,3)
plt.hist(data.Average_Kernel_Mass)
plt.title('Kernel Mass\n(20k Avg)')
plt.subplot(2,3,4)
plt.hist(data.Prop_Retained)
plt.title('Proportion Retained')
plt.subplot(2,3,5)
plt.hist(data.Initial_Pericarp_Prop)
plt.title('Proportion Pericarp')
plt.subplot(2,3,6)
plt.hist(data.Prop_Retained_Norm)
plt.title('Norm Proportion Pericarp')
plt.tight_layout()
plt.show()

# Write dataset out for debugging
#data.to_excel('~/Desktop/Grad_School/Research/Projects/Pericarp/Combined_Datasets.xlsx', index = False)

# How well does pericarp retention correlate with initial pericarp level?
model = np.polyfit(data.Average_Pericarp_Initial, data.Prop_Retained, 1)
p = np.poly1d(model)
plt.scatter(data.Average_Pericarp_Initial, data.Prop_Retained, c = 'black')
plt.plot(data.Average_Pericarp_Initial, p(data.Average_Pericarp_Initial), c = 'black')
plt.text(x = 18, y = 0.15, s = 'R = ' + str(round(pearsonr(data.Average_Pericarp_Initial,
                                                          data.Prop_Retained)[0],
                                                 3)))
plt.text(x = 18, y = 0.05, s = 'P = ' + str(round(pearsonr(data.Average_Pericarp_Initial,
                                                          data.Prop_Retained)[1],
                                                 3)))
plt.ylim(0,1)
plt.title('Mass of Pericarp on Raw Kernel')
plt.xlabel('')
plt.ylabel('Pericarp Retained')
plt.show()
# Initial pericarp content doesn't seem to play a role in pericarp retention
# this might mean that it is largely compositionally based?
# Check to see if there is a correlation between the pericarp retained and
# initial pericarp content normalized by kernel size.

# What if we normalize initial pericarp by the kernel mass?
model = np.polyfit(data.Initial_Pericarp_Prop, data.Prop_Retained, 1)
p = np.poly1d(model)
plt.scatter(data.Initial_Pericarp_Prop, data.Prop_Retained, c = 'black')
plt.plot(data.Initial_Pericarp_Prop, p(data.Initial_Pericarp_Prop), c = 'black')
plt.text(x = 0.07, y = 0.15, s = 'R = ' + str(round(pearsonr(data.Initial_Pericarp_Prop,
                                                          data.Prop_Retained)[0],
                                                 3)))
plt.text(x = 0.07, y = 0.05, s = 'P = ' + str(round(pearsonr(data.Initial_Pericarp_Prop,
                                                          data.Prop_Retained)[1],
                                                 3)))
plt.ylim(0,1)
plt.xlabel('')
plt.ylabel('')
plt.title('Pericarp Corrected by Kernel Mass')
plt.yticks([])
plt.show()
# There is a slight correlation here.  Maybe it has to do with compositional
# resource sinks?  Or maybe it better relates to the idea that with larger
# kernels you have a larger surface area have pericarp on and so you need to
# consider that when dealing with pericarp retention?  In theory there could
# an effect of surface area:depth or surface area:volume ratio?

# How well does pericarp retention correlate with kernel mass?
model = np.polyfit(data.Average_Kernel_Mass, data.Prop_Retained, 1)
p = np.poly1d(model)
plt.scatter(data.Average_Kernel_Mass, data.Prop_Retained)
plt.plot(data.Average_Kernel_Mass, p(data.Average_Kernel_Mass))
plt.text(x = 100, y = 0.15, s = 'R = ' + str(round(pearsonr(data.Average_Kernel_Mass,
                                                          data.Prop_Retained)[0],
                                                 3)))
plt.text(x = 100, y = 0.1, s = 'P = ' + str(round(pearsonr(data.Average_Kernel_Mass,
                                                          data.Prop_Retained)[1],
                                                 3)))
plt.xlabel('Kernel Mass')
plt.ylabel('Proportion of Pericarp Retained')
plt.show()
# Kernel mass and pericarp retention seem to be inversely correlated, which
# is not very surprising.  As the kernel gets larger, so does the surface area
# and potential places for chemical reactions to occur.  This means that the
# pericarp will have more holes that can expand, and thus will lead to less
# pericarp on the kernel after cooking (i.e. lower pericarp retention).

# How well do NIR fiber predictions correlate with pericarp retention?  
# Read in NIR Scan Data (Note: There is only data for the commercial hybrids right now (N = 20))
scans = pd.read_csv('~/Desktop/Grad_School/Research/Projects/Pericarp/Data/Commercial_Hybrids_Pericarp_Retention_and_Proximates.csv')


# Plot correlation between fiber prediction and proportion pericarp retained
model = np.polyfit(scans['Fiber'], scans.Pericarp_Retained, 1)
p = np.poly1d(model)
plt.scatter(scans['Fiber'], scans.Pericarp_Retained, c = 'black')
plt.plot(scans['Fiber'], p(scans['Fiber']), c = 'black')
plt.text(x = 1.82, y = 0.07, s = 'R = ' + str(round(pearsonr(scans['Fiber'],
                                                           scans.Pericarp_Retained)[0],
                                                 3)))
plt.text(x = 1.82, y = 0.02, s = 'P = ' + str(round(pearsonr(scans['Fiber'],
                                                            scans.Pericarp_Retained)[1],
                                                 3)))
plt.xlabel('')
plt.ylabel('')
plt.title('NIR Predicted Fiber Content')
plt.ylim(0,1)
plt.yticks([])
plt.show()

# How well does initial pericarp correlate with NIR fiber predictions?
model = np.polyfit(scans['Fiber As is'], scans.Average_Pericarp_Initial, 1)
p = np.poly1d(model)
plt.scatter(scans['Fiber As is'], scans.Average_Pericarp_Initial)
plt.plot(scans['Fiber As is'], p(scans['Fiber As is']))
plt.text(x = 1.2, y = 11.5, s = 'R = ' + str(round(pearsonr(scans['Fiber As is'],
                                                           scans.Average_Pericarp_Initial)[0],
                                                 3)))
plt.text(x = 1.2, y = 11, s = 'P = ' + str(round(pearsonr(scans['Fiber As is'],
                                                            scans.Average_Pericarp_Initial)[1],
                                                 3)))
plt.xlabel('Fiber Content Prediction')
plt.ylabel('Initial Pericarp Content')
plt.show()

# What if we normalize the plots above for kernel mass?
# Plot correlation between fiber prediction and proportion pericarp retained normalized by kernel mass
model = np.polyfit(pericarp_and_scans['Fiber As is'], pericarp_and_scans.Prop_Retained_Norm, 1)
p = np.poly1d(model)
plt.scatter(pericarp_and_scans['Fiber As is'], pericarp_and_scans.Prop_Retained_Norm)
plt.plot(pericarp_and_scans['Fiber As is'], p(pericarp_and_scans['Fiber As is']))
plt.text(x = 1.2, y = 0.0004, s = 'R = ' + str(round(pearsonr(pericarp_and_scans['Fiber As is'],
                                                           pericarp_and_scans.Prop_Retained_Norm)[0],
                                                 3)))
plt.text(x = 1.2, y = 0.0003, s = 'P = ' + str(round(pearsonr(pericarp_and_scans['Fiber As is'],
                                                            pericarp_and_scans.Prop_Retained_Norm)[1],
                                                 3)))
plt.xlabel('Fiber Content Prediction')
plt.ylabel('Proportion of Pericarp Retained / Kernel Mass')
plt.show()

# How well does initial pericarp normalized for kernel mass correlate with NIR fiber predictions?
model = np.polyfit(pericarp_and_scans['Fiber As is'], pericarp_and_scans.Initial_Pericarp_Prop, 1)
p = np.poly1d(model)
plt.scatter(pericarp_and_scans['Fiber As is'], pericarp_and_scans.Initial_Pericarp_Prop)
plt.plot(pericarp_and_scans['Fiber As is'], p(pericarp_and_scans['Fiber As is']))
plt.text(x = 1.2, y = 0.032, s = 'R = ' + str(round(pearsonr(pericarp_and_scans['Fiber As is'],
                                                           pericarp_and_scans.Initial_Pericarp_Prop)[0],
                                                 3)))
plt.text(x = 1.2, y = 0.031, s = 'P = ' + str(round(pearsonr(pericarp_and_scans['Fiber As is'],
                                                            pericarp_and_scans.Initial_Pericarp_Prop)[1],
                                                 3)))
plt.xlabel('Fiber Content Prediction')
plt.ylabel('Initial Pericarp Content / Kernel Mass')
plt.show()

# What about Protein? How does it correlate?
# Plot correlation between protein prediction and proportion pericarp retained
model = np.polyfit(scans['Protein'], scans.Pericarp_Retained, 1)
p = np.poly1d(model)
plt.scatter(scans['Protein'], scans.Pericarp_Retained, c = 'black')
plt.plot(scans['Protein'], p(scans['Protein']), c = 'black')
plt.text(x = 7, y = 0.07, s = 'R = ' + str(round(pearsonr(scans['Protein'],
                                                           scans.Pericarp_Retained)[0],
                                                 3)))
plt.text(x = 7, y = 0.02, s = 'P = ' + str(round(pearsonr(scans['Protein'],
                                                            scans.Pericarp_Retained)[1],
                                                 3)))
plt.xlabel('')
plt.ylabel('')
plt.title('NIR Predicted Protein Content')
plt.ylim(0,1)
plt.yticks([])
plt.show()

# How well does initial pericarp correlate with NIR protein predictions?
model = np.polyfit(pericarp_and_scans['Protein As is'], pericarp_and_scans.Average_Pericarp_Initial, 1)
p = np.poly1d(model)
plt.scatter(pericarp_and_scans['Protein As is'], pericarp_and_scans.Average_Pericarp_Initial)
plt.plot(pericarp_and_scans['Protein As is'], p(pericarp_and_scans['Protein As is']))
plt.text(x = 4, y = 11.5, s = 'R = ' + str(round(pearsonr(pericarp_and_scans['Protein As is'],
                                                           pericarp_and_scans.Average_Pericarp_Initial)[0],
                                                 3)))
plt.text(x = 4, y = 11, s = 'P = ' + str(round(pearsonr(pericarp_and_scans['Protein As is'],
                                                            pericarp_and_scans.Average_Pericarp_Initial)[1],
                                                 3)))
plt.xlabel('Protein Content Prediction')
plt.ylabel('Initial Pericarp Content')
plt.show()

# What if we normalize the plots above for kernel mass?
# Plot correlation between protein prediction and proportion pericarp retained normalized by kernel mass
model = np.polyfit(pericarp_and_scans['Protein As is'], pericarp_and_scans.Prop_Retained_Norm, 1)
p = np.poly1d(model)
plt.scatter(pericarp_and_scans['Protein As is'], pericarp_and_scans.Prop_Retained_Norm)
plt.plot(pericarp_and_scans['Protein As is'], p(pericarp_and_scans['Protein As is']))
plt.text(x = 4, y = 0.0004, s = 'R = ' + str(round(pearsonr(pericarp_and_scans['Protein As is'],
                                                           pericarp_and_scans.Prop_Retained_Norm)[0],
                                                 3)))
plt.text(x = 4, y = 0.0003, s = 'P = ' + str(round(pearsonr(pericarp_and_scans['Protein As is'],
                                                            pericarp_and_scans.Prop_Retained_Norm)[1],
                                                 3)))
plt.xlabel('Protein Content Prediction')
plt.ylabel('Proportion of Pericarp Retained / Kernel Mass')
plt.show()

# How well does initial pericarp normalized for kernel mass correlate with NIR protein predictions?
model = np.polyfit(pericarp_and_scans['Protein As is'], pericarp_and_scans.Initial_Pericarp_Prop, 1)
p = np.poly1d(model)
plt.scatter(pericarp_and_scans['Protein As is'], pericarp_and_scans.Initial_Pericarp_Prop)
plt.plot(pericarp_and_scans['Protein As is'], p(pericarp_and_scans['Protein As is']))
plt.text(x = 4, y = 0.032, s = 'R = ' + str(round(pearsonr(pericarp_and_scans['Protein As is'],
                                                           pericarp_and_scans.Initial_Pericarp_Prop)[0],
                                                 3)))
plt.text(x = 4, y = 0.031, s = 'P = ' + str(round(pearsonr(pericarp_and_scans['Protein As is'],
                                                            pericarp_and_scans.Initial_Pericarp_Prop)[1],
                                                 3)))
plt.xlabel('Protein Content Prediction')
plt.ylabel('Initial Pericarp Content / Kernel Mass')
plt.show()