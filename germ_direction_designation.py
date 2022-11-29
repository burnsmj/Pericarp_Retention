#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:22:30 2022

@author: michael
"""
### Germ Direction Designation
### Michael Burns
### 11/10/22

# Purpose:  To create a record of germ direction for the kernel images.  This
#           was done by the undergraduates, but after having to correct 2 in
#           the first 10 images I decided it would be better to redo the images.
#           I thought this would be easiest in an interactive python script
#           where the script will show an image and record the designation as
#           showing germ (y) or not showing germ (n) along with the file name.
#           apart from a little bit of time coding, it should be faster than
#           flipping between pictures and excel.

#           Before starting this, a file of junk images should be created to
#           minimize extra work on this.  By knowing which images are bad, I
#           can skip kernels from images that don't show quality staining.

def germ_direction_designation(images):
    