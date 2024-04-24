#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:22:02 2022

@author: michael
"""
### Finding Plots that Segregate for Fiber
### Michael Burns
### 3/25/22

###################
# Package Imports #
###################
import pandas as pd
import numpy as np

################################
# Read Excel File of NIR Scans #
################################
nir_scans_raw = pd.read_excel('../EMS_BSA/Data/NIR_Scans/Global_Equations_Modified.xlsx')

nir_scans_raw.ID = nir_scans_raw.ID.str.upper()

nir_scans_filtered = nir_scans_raw[nir_scans_raw.ID.str.startswith('YC', na = False)]

print(nir_scans_filtered)

######################################
# Remove samples that needed rescans #
######################################
rescan_samples = []
for sample in nir_scans_filtered.ID:
    if 'RESCAN' in sample:
        rescanned_id = sample.split('_')[0]
        rescan_samples.append(rescanned_id)

print(rescan_samples)

nir_scans_filtered = nir_scans_filtered[~nir_scans_filtered.ID.isin(rescan_samples)]

nir_scans_filtered.ID = nir_scans_filtered.ID.str.rstrip('_RESCAN')

print(nir_scans_filtered)

nir_fiber = nir_scans_filtered[['ID', 'Fiber As is']].sort_values(by = 'Fiber As is')
print(nir_fiber)

median_fiber = np.median(nir_fiber['Fiber As is'])
q25_fiber, q75_fiber = np.percentile(nir_fiber['Fiber As is'], [25,75])
iqr_fiber = q75_fiber - q25_fiber
print(median_fiber)
print(iqr_fiber)

lower_thresh_fiber = q25_fiber - (1.5 * iqr_fiber)
upper_thresh_fiber = q75_fiber + (1.5 * iqr_fiber)

print(lower_thresh_fiber)
print(upper_thresh_fiber)

nir_fiber_clean = nir_fiber[nir_fiber['Fiber As is'] > lower_thresh_fiber][nir_fiber['Fiber As is'] < upper_thresh_fiber]

yc_ids = pd.read_csv('../EMS_BSA/Data/YC_ID_Decoder.csv').iloc[:,[1,3]].rename(columns = {'Plot' : 'ID'})

geno_fiber = nir_fiber_clean.merge(yc_ids)

geno_fiber_mean = geno_fiber.groupby('Genotype').mean('Fiber As is').sort_values('Fiber As is')

nir_fiber_samples = geno_fiber_mean.iloc[::37, :] # 55 is the maximum number allowed to collect 10 samples from the dataset as is.  I wanted the maximum number to get the maximum range of values.
print(nir_fiber_samples)
print(len(nir_fiber_samples))
