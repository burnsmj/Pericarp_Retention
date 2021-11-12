#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 09:15:03 2021

@author: michael
"""
### Training Data Tabulation
### Michael Burns
### 9/27/21

#def Training_Data_Tabulation(dataset, corn_color = 'white'): # Uncomment this if a function is desired.  At the time of making this, it didnt seem that important.  If you create a function with this, make sure to change the filepath that is read in during the loop below.

"""
The purpose of this script is to tabulate the training data collected in 
ImageJ.  This data comes in the form of a plain text file with section 
classifications in a header row starting with an octothorpe.  All pixel values
are pasted below the header line in R,G,B format and separated by tabs.  This 
script should use the section header as a column header, and turn each section
into a column of comma separated values.
"""

###################
# Import Packages #
###################
import pandas as pd # Data wrangling and manipulation

############################
# Create Lists to Populate #
############################
pericarp = []
endosperm = []
background = []

#############################
# Read in Data Line by Line #
#############################
with open('../Data/White_Pixel_Values_for_Training.txt') as imagej_data: # Load data file path to read line by line
    lines = imagej_data.readlines() # Read in dataset line by line
    for line in lines: # Loop to iterate through lines
        #print('Line:', line) # For possible future debugging; Print the line we are working on
        if line.startswith('#'): # If the line starts with a #, save it as the variable we will use to separate the data
            current_header = line.strip() # need to strip the line of whitespaces before comparing it (likely due to new line character)
        elif current_header == '#Pericarp': # For the pericarp data...
            line_items = line.split() # Split the values by whitespace
            pericarp += line_items # Add the data to the pericarp list
        elif current_header == '#Aleurone': # For the pericarp data...
            line_items = line.split() # Split the values by whitespace
            endosperm += line_items # Add the data to the pericarp list
        elif current_header == '#Background': # For the pericarp data...
            line_items = line.split() # Split the values by whitespace
            background += line_items # Add the data to the pericarp list
        else:
            print('Error: Data does not contain the required section headers: #pericarp, #endosperm, #background')
            break

assert len(pericarp) == len(endosperm) and len(pericarp) == len(background) # Make sure each dataset is the same size. This can be removed later if its better to collect more of one dataset than another.

#print(pericarp) # For possible future debugging
#print(endosperm) # For possible future debugging
#print(background) # For possible future debugging


###########################
# Turn Lists into Columns #
###########################
tab_data = pd.DataFrame(data = [pericarp, endosperm, background]).transpose() # Turn list into dataframe columns and transpose to make 3 columns
tab_data.columns = ['Pericarp', 'Aleurone', 'Background'] # Add headers to each column

###############################
# Write out Tabulated Dataset #
###############################
tab_data.to_csv('../Data/White_Corn_Tabulated_Training_Data.txt', sep = '\t', index = False)
