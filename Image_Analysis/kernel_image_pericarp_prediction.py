#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:01:34 2022

@author: michael
"""
# Kernel Image Pericarp Prediction
# Michael Burns
# 8/24/22

# Purpose:  create a CNN model that will learn the pattern of pericarp
#           labeled in images (hopefully by color and texture) and predict to
#           a high degree of accuracy the pericarp areas on new kernel images.
#           These predictions can then be used calculate pericarp coverage,
#           intensity and quantity.  This can then be correlated with ground
#           truth data that has already been collected.

# Expected Steps:   1.) Import packages
#                   2.) Read in images
#                   3.) Read in annotations
#                   4.) Create masks
#                   5.) Downsample images and their masks
#                           May need to perform some image modifications as
#                           well to increase training pool size
#                   6.) Set aside the 100 common images that everyone annotated
#                       for evaluation of the model
#                   6.) Train a model (CNN)
#                   7.) Test the model on the 100 common images by correlating
#                       Coverage to each of the annotators
#                   8.) Calculate Accuracy of prediction for the 100 common
#                       images for each of the annotators to the model
#                   9.) Determine pericarp intensity
#                   10.)Determine pericarp quantity
#                   11.)Correlate values with ground truth data
