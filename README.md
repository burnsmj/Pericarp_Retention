# Maize kernel composition and morphology influences pericarp retention during nixtamalization
by Michael J. Burns<sup>1</sup>, Sydney Berry<sup>1</sup>, Molly Loftus<sup>1</sup>, Amanda M. Gilbert<sup>1</sup>, Peter J. Hermanson<sup>1</sup>, and Candice N. Hirsch<sup>1</sup>  
<sup>1</sup>Department of Agronomy and Plant Genetics, University of Minnesota

## Purpose:
Food-grade maize is an understudied segment of maize research due to limited production acreage across to United
States. This has led to a reduced understanding the maize kernel features that influence the final product of
nixtamalized goods such as tortillas or tortilla chips. One kernel aspect that impacts many downstream product
quality traits in the quantity of pericarp retained after nixtamalization. The quantity of pericarp retained can
impact the amount of moisture absorbed, the cohesion and adhesion of the dough, and the quantity of fiber present
in the final product. Previous research has proposed a high throughput method to quantify the area of kernel covered
by pericarp through boiling kernels in an alkaline solution and stained kernels with a methlyene blue solution to
increase contrast. This method only tested the kernel area coverage, and did not consider the third dimension of
pericarp retention: the depth of the pericarp retained. This work aims to build upon the previous method by further
optimizing the staining protocol and investigating which compositional and morphological components of a maize kernel
influence the quantity of pericarp retained.

## Publication:
For more information, please see the manuscript at: **ADD DOI HERE**

## Pipeline Description:
This repository focused on characterizing pericarp retention in maize hybrids using compositional, spectral, and
morphological data. The experimental panel consisted of 819 hybrids derived from crosses between three pollen donors
and ~280 inbred lines. Field trials were conducted in 2022 and 2023 in a randomized complete block design. Grain from
each plot was analyzed using two near-infrared (NIR) spectrometers, Perten DA7250 (ground grain) and FOSS Infratec Nova
(whole kernel), to obtain spectral and compositional profiles.

Spectrally diverse samples were selected using Honigs regression on preprocessed FOSS spectra to capture variation across
pollen donors and years. These samples were evaluated using both rapid and benchtop nixtamalization methods. Pericarp
retention was quantified through two approaches: (1) manual pericarp removal and dry mass measurement and (2) an optimized
methylene blue staining protocol followed by visual scoring. Kernel morphology traits were extracted from high-resolution
images and supplemented with manual measurements (mass, volume, density, and initial pericarp quantity).

The relationship between pericarp retention and grain traits was assessed using correlation analyses and predictive modeling.
A lasso-based elastic net model was built with compositional and morphological features to identify key predictors of pericarp
retention. Final trait associations were validated in independent hybrid subsets using benchtop cook tests and statistical
comparisons, providing insight into the physical and biochemical drivers of pericarp adherence during processing.

## Directories:
Chemometrics
- Code developed to understand biological relationship of kernel composition and shape with pericarp retention.
- NIR spectra was collected with a FOSS Infratec NOVA, using the STM system with the cuvette pathlength set to 29mm.
- Kernel shape/morphology was collected through image analysis or by hand (initial pericarp quantity, volume, mass, density)

Image Analysis
 - Code developed to assess images of stained kernels. This direction was discontinued, but code is still provided.
 - Code includes kernel segmentation, color property extraction, classification, and more.
