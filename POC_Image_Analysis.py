from plantcv import plantcv as pcv

# Run naive bayes multiclass and save a list of masks
mask = pcv.naive_bayes_classifier(img, pdf_file="plantcv/naive_bayes_pdfs.txt")

# Plot each class with it's own color
plotted = pcv.visualize.colorize_masks(masks=[mask['pericarp'], mask['endosperm'], mask['background']],
                                       colors=['green', 'red', 'gray'])