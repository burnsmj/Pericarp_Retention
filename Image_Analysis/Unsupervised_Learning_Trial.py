### Unsupervised Learning of Kernel Images
### Michael Burns
### 9/14/21

###########
# Purpose #
###########
"""
    The purpose of this script is to try unsupervised learning methods on the 
images of kernels that Mark took.  If this method shows promise, it could be 
used on future data.  The hope is to let a model determine the best 3 
clusters of data and plot them for visualization.  If they seem to follow
a pattern similar to the actual pericarp pattern on the kernels, we could
just use this model to determine good parameters.
    
    An additional purpose of this script is to learn more about unsupervised 
learning, how to code in python, and what machine learning is like in 
python.
"""

#########
# Notes #
#########
"""
    At the time of creation, Image_Processing.py was already created and run, 
so for simplicity and to reduce redundancy on something that may not work, 
I will be accessing the variables created in that script here.
"""

#####################
# Required Packages #
#####################
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


################
# Get the Data #
################
from Image_Processing import Pixel_Data
#print(Pixel_Data)

######################
# K Means Clustering #
######################
kmeans_model = KMeans(n_clusters = 3)

kmeans_model.fit(Pixel_Data[['Red','Green','Blue',]])

Predictions = kmeans_model.predict(Pixel_Data[['Red','Green','Blue']])

Pixel_Data.insert(2, 'Prediction', Predictions)

Coordinated_Predictions = Pixel_Data.iloc[:,0:3]
print(Coordinated_Predictions)

print('Unsupervised Pixel Classification:')
print(Pixel_Data)

##########################
# Pivot Pixel Data Wider #
##########################
Wide_Predictions = Coordinated_Predictions.pivot(index = 'Row', columns = 'Column', values = 'Prediction')
print(Wide_Predictions)


##########################
# Plot Pixel Predictions #
##########################
plt.imshow(Wide_Predictions)
plt.show()







