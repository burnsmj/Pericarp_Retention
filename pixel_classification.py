#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:59:31 2021

@author: michael
"""
### Pixel Classification
### Michael Burns
### 10/27/21

#####################################################################################
# Probably not going to be used in the long run, but worth keeping for example code #
#####################################################################################

# Purpose: To learn about creating predictive algorithms in python
#          To predict the classes of a pixel based on training data
#          To determine which model-parameter combo is best for my image data


####################
# Import Libraries #
####################
import numpy as np
import pandas as pd
import image_kernel_detection
import time
import glob
from random import seed
from skimage import io
from skimage import filters
from skimage import util
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
#from sklearn.metrics import classification_report
from sklearn import naive_bayes # Base level model approach
from sklearn import svm # Complex, accurate, slow approach
from sklearn import neighbors # Clustering approach
from sklearn import tree # Decision tree approach
from sklearn import ensemble # Multimodel approach
from sklearn import linear_model # Neural network approach
import xgboost as xgb
import matplotlib.pyplot as plt

#################
# Read in Image #
#################
image_rgb = io.imread('../Data/Images/Trials/Poststain_White_MB_H2O.png')[:,:,0:3]

##############
# Blur Image #
##############
image_blur = util.img_as_ubyte(filters.gaussian(image_rgb,
                                                sigma = 5,
                                                multichannel = True,
                                                truncate = 2))

###########################
# Create Validation Image #
###########################
valid_image = image_kernel_detection.image_background_removal(image_blur)

#########################
# Read in Training Data #
#########################
pixel_data = pd.read_table('../Data/White_Corn_Tabulated_Training_Data.txt',
                           delimiter = '\t')

###############################################
# Remove Background Pixels from Training Data #
###############################################
RM_BKG_PIX = True

if RM_BKG_PIX is True:
    pixel_data = pixel_data[pixel_data.Label != 'Background']

###########################
# Encoding Labels as Ints #
###########################
le = LabelEncoder()
pixel_data.loc[:, 'Label_Codes'] = le.fit_transform(pixel_data['Label'])

###########################
# Splitting Training Data #
###########################
train, test = train_test_split(pixel_data,
                               test_size = 0.8, 
                               random_state = 7)

#####################################
# Plotting Training Set Information #
#####################################
##############################################################
# Plot Distributions of Each Channel for Each Classification #
##############################################################
def training_data_distributions(train_data,
                                channel_names = ('Red', 'Green', 'Blue',
                                                 'Hue', 'Saturation', 'Value',
                                                 'Luminosity', 'A_Axis', 'B_Axis'),
                                alpha = 0.7):
    n_row = int(len(channel_names) / 3)
    plot_num = 1

    for channel in channel_names:
        plt.subplot(n_row,3,plot_num)
        plt.hist(train_data[channel][train_data.Label_Codes == 2],
                 color = 'black',
                 alpha = alpha,
                 bins = 50)
        plt.hist(train_data[channel][train_data.Label_Codes == 1],
                 color = 'blue',
                 alpha = alpha,
                 bins = 50)
        plt.hist(train_data[channel][train_data.Label_Codes == 0],
                 color = 'yellow',
                 alpha = alpha,
                 bins = 50)
        plt.title(channel + ' Distribution')

        plot_num += 1
    
    plt.tight_layout()
    plt.show()

training_data_distributions(train)

#####################################################
# Plot Combinations of Each Channel for Color Space #
#####################################################
def training_data_combinations(train_data,
                               channel_names = [('Red', 'Green', 'Blue'),
                                                ('Hue', 'Saturation', 'Value'),
                                                ('Luminosity', 'A_Axis', 'B_Axis')],
                               colors = {'Aleurone' : 'yellow',
                                         'Pericarp' : 'blue',
                                         'Light_Pericarp' : 'green',
                                         'Dark_Pericarp' : 'black',
                                         'Background' : 'purple'},
                               alpha = 0.2):
    n_row = int(len(channel_names))
    plot_num = 1

    for channel_group in channel_names:
        channel_num_1 = 0
        channel_num_2 = 1
        
        for channel in channel_group:
            plt.subplot(n_row,3,plot_num)
            plt.scatter(train_data[channel_group[channel_num_1]], 
                        train_data[channel_group[channel_num_2]],
                        c = train_data.Label.map(colors),
                        alpha = alpha,
                        marker = '.')
            plt.xlabel(channel_group[channel_num_1])
            plt.ylabel(channel_group[channel_num_2])
            
            plot_num += 1
            channel_num_1 += 1
            channel_num_2 += 1
            
            if(channel_num_1 == 2):
                channel_num_2 = 0

    plt.tight_layout()
    plt.show()
    
training_data_combinations(train)

#################################################
# Plot Heatmap of Correlations Between Features #
#################################################
def training_data_correlations(train_data, show_num = False):
    plt.imshow(train_data.corr() ** 2, cmap = 'hot', interpolation = 'nearest')
    plt.xticks(ticks = range(0,10),
               labels = train_data.columns.values.tolist()[1:],
               rotation = -45,
               ha = 'left',
               rotation_mode = 'anchor')
    plt.yticks(ticks = range(0,10),
               labels = train_data.columns.values.tolist()[1:])
    plt.colorbar()
    plt.title('PVE Between Features and Label')
    plt.show()
    
    if show_num is True:
        print('Training Data PVE:\n', train_data.corr() ** 2)
    
training_data_correlations(train, show_num = True)

#########################################
# Initial Testing of Model Performances #
#########################################
def train_many_models(train_data, 
                     models_list = [('NB', 
                                     naive_bayes.GaussianNB(), 
                                     {}),
                                    ('KNN', 
                                     neighbors.KNeighborsClassifier(), 
                                     {'n_neighbors' : (3, 5, 7, 9),
                                      'weights' : ('uniform',
                                                   'distance')}),
                                    ('DT',
                                     tree.DecisionTreeClassifier(),
                                     {'criterion' : ('gini',
                                                     'entropy')}),
                                    #('LOGIT',
                                    # linear_model.LogisticRegression(),
                                    # {'penalty' : ('l1',
                                    #               'l2'),
                                    #  'C' : (1, 10, 50, 100),
                                    #  'fit_intercept' : (True,
                                    #                     False)}),
                                    ('SVC',
                                     svm.SVC(),
                                     {'C' : (1, 10, 50, 100, 500, 1000),
                                      'kernel' : ('linear',
                                                  'rbf',
                                                  'sigmoid')}),
                                    ('RF',
                                     ensemble.RandomForestClassifier(),
                                     {'criterion' : ('gini',
                                                     'entropy'),
                                                                               'n_estimators' : (3, 25, 50, 75, 100)}),
                                    ('XGB',
                                     xgb.XGBClassifier(eval_metric = 'mlogloss',
                                                              use_label_encoder = False),
                                     {'learning_rate' : (0.1, 0.3, 0.5, 0.7, 0.9),
                                      'booster' : ('gbtree',
                                                   'gblinear',
                                                   'dart')}),
                                    ('RC',
                                     linear_model.RidgeClassifier(),
                                     {'alpha' : (0.5, 1, 1.5, 2, 3, 5),
                                      'fit_intercept' : (True,
                                                         False),
                                      'normalize' : (True,
                                                     False)})],
                     features = ['Red', 'Green', 'Blue',
                                'Hue', 'Saturation', 'Value',
                                'Luminosity', 'A_Axis', 'B_Axis']):

    ############################
    # Separating Training Data #
    ############################
    train_x = train_data[features]
    train_y = train_data['Label_Codes']
    
    #################
    # Storage Lists #
    #################
    best_models = {}

    for name, model, params in models_list:
        seed(7)
        start = time.time()
        grid_cv = GridSearchCV(estimator = model,
                               cv = 10,
                               param_grid = params,
                               scoring = 'accuracy')
        grid_cv.fit(train_x,
                    train_y)
        end = time.time()

        if grid_cv.best_estimator_ not in best_models:
            best_models[grid_cv.best_estimator_] = {'Time' : end - start,
                                                    'Avg_Accuracy' : grid_cv.cv_results_['mean_test_score'][grid_cv.best_index_],
                                                    'Var_Accuracy' : grid_cv.cv_results_['std_test_score'][grid_cv.best_index_]}
    
    return best_models

###################################
# Learning Curves for Many Models #
###################################
def learning_curves(train_data,
                    model,
                    features = ['Red', 'Green', 'Blue',
                               'Hue', 'Saturation', 'Value',
                               'Luminosity', 'A_Axis', 'B_Axis'],
                    train_sizes = np.linspace(0.005,1.0,20),
                    n_cv = 10):
    ############################
    # Separating Training Data #
    ############################
    train_x = train_data[features]
    train_y = train_data['Label_Codes']
    
    seed(7)
    
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model,
                                                                          train_x,
                                                                          train_y,
                                                                          train_sizes = train_sizes,
                                                                          cv = n_cv,
                                                                          return_times = True)
    
    return train_sizes, train_scores, test_scores, fit_times

def test_model_on_image(train_data,
                        model,
                        image,
                        features = ['Red', 'Green', 'Blue',
                                   'Hue', 'Saturation', 'Value',
                                   'Luminosity', 'A_Axis', 'B_Axis'],
                        give_image = True,
                        give_coverage = True):
    test_image = np.zeros(image.shape[0:2])

    train_x = train_data[features]
    train_y = train_data['Label_Codes']

    model.fit(train_x,
              train_y)
    
    results = []
    
    for kernel in range(1, image_kernel_detection.image_labelling(image).max() + 1):
        test_tab = image_kernel_detection.indiv_kernel_data_extraction(image,
                                                                  kernel_number = kernel)

        preds = model.predict(test_tab[features])
        
        results.append(round((sum(le.inverse_transform(preds) == 'Aleurone') / test_tab.shape[0]) * 100, 2))
        
        x_data = np.array(test_tab.X).astype(int)
        y_data = np.array(test_tab.Y).astype(int)
        
        test_image[y_data, x_data] = preds.transpose() + 1

    if give_coverage is True and give_image is True:
        return test_image, np.array(results)
    elif give_coverage is True and give_image is False:
        return np.array(results)
    elif give_coverage is False and give_image is True:
        return test_image
    else:
        print('No outputs were requested.\nPlease set give_image or give_coverage to True')

####################################################################
# Display Many Combinations of Training Sets, Features, and Images #
####################################################################
train_sets = sorted(glob.glob('../Data/White_Corn_Tabulated*Training_Data.txt'))
print(train_sets)

images = sorted(glob.glob('../Data/Images/Trials/Multi_Genotype_Images/*W*Grid*.tif'))
print(images)

features_list = [
                 ['Red', 'Green', 'Blue',
                  'Hue', 'Saturation', 'Value',
                  'Luminosity', 'A_Axis', 'B_Axis'],
                 ['Hue', 'Saturation', 'A_Axis', 'B_Axis'],
                 #['Red', 'Green', 'Blue'],
                 #['Hue', 'Saturation', 'Value'],
                 #['Luminosity', 'A_Axis', 'B_Axis']
                 ]

models_list = [('NB', 
                naive_bayes.GaussianNB(), 
                {}),
               ('KNN', 
                neighbors.KNeighborsClassifier(), 
                {'n_neighbors' : (3, 5, 7, 9),
                 'weights' : ('uniform',
                              'distance')}),
               ('DT',
                tree.DecisionTreeClassifier(),
                {'criterion' : ('gini',
                                'entropy')}),
               #('LOGIT',
               # linear_model.LogisticRegression(),
               # {'penalty' : ('l1',
               #               'l2'),
               #  'C' : (1, 10, 50, 100),
               #  'fit_intercept' : (True,
               #                     False)}),
               ('SVC',
                svm.SVC(),
                {'C' : (1, 10, 50, 100, 500, 1000),
                 'kernel' : ('linear',
                             'rbf',
                             'sigmoid')}),
               ('RF',
                ensemble.RandomForestClassifier(),
                {'criterion' : ('gini',
                                'entropy'),
                                                          'n_estimators' : (3, 25, 50, 75, 100)}),
               ('XGB',
                xgb.XGBClassifier(eval_metric = 'mlogloss',
                                         use_label_encoder = False),
                {'learning_rate' : (0.1, 0.3, 0.5, 0.7, 0.9),
                 'booster' : ('gbtree',
                              'gblinear',
                              'dart')}),
               ('RC',
                linear_model.RidgeClassifier(),
                {'alpha' : (0.5, 1, 1.5, 2, 3, 5),
                 'fit_intercept' : (True,
                                    False),
                 'normalize' : (True,
                                False)})]

for model in models_list:
    print(model)
    plot_num = 1

    plt.figure(figsize = (8,16), dpi = 300)

    for image in images:
        print(image)

        input_image = io.imread(image, plugin = 'pil')

        for features in features_list:
            if(plot_num not in [1, 6, 11, 16, 21]):
                print('Working on plot: ' + str(plot_num))
    
                plt.subplot(5,5,plot_num)
                plt.imshow(input_image)
                plt.axis('off')
                if plot_num == 3:
                    plt.title('Original')
    
                plot_num += 1

            for train in train_sets:
                print('Working on plot: ' + str(plot_num))
                # Read in Training Data #
                pixel_data = pd.read_table(train,
                                           delimiter = '\t')

                # Remove Background Pixels from Training Data #
                pixel_data = pixel_data[pixel_data.Label != 'Background']

                # Encoding Labels as Ints #
                le = LabelEncoder()
                pixel_data.loc[:, 'Label_Codes'] = le.fit_transform(pixel_data['Label'])

                # Splitting Training Data #
                train, test = train_test_split(pixel_data,
                                               test_size = 0.8, 
                                               random_state = 7)

                current_model = GridSearchCV(estimator = model[1],
                                             cv = 10,
                                             param_grid = model[2],
                                             scoring = 'accuracy')

                current_model.fit(train[features],
                                  train['Label_Codes'])

                predicted_image = test_model_on_image(train,
                                                      current_model,
                                                      input_image,
                                                      features = features,
                                                      give_image = True,
                                                      give_coverage = False)

                plt.subplot(5,5,plot_num)
                plt.imshow(predicted_image)
                plt.axis('off')

                if plot_num == 1 or plot_num == 2:
                    plt.title('RGBHSVLAB')
                if plot_num == 4 or plot_num == 5:
                    plt.title('HSAB')

                plot_num += 1
    plt.suptitle(model[0], y = 0.92)
    plt.savefig('junk_test.png')





####################################################
# Try the Best Model on Each of the Trial Pictures #
####################################################
VALIDATE_MODEL = False

if VALIDATE_MODEL is True:
    

    chosen_model = svm.SVC(C = 1)

    chosen_model.fit(train[['Red', 'Green', 'Blue',
                            'Hue', 'Saturation', 'Value',
                            'Luminosity', 'A_Axis', 'B_Axis']],
                     train['Label_Codes'])

    images = sorted(glob.glob('../Data/Images/Trials/Multi_Genotype_Images/*W*.tif'))
    print(images)

    for image in images:
        input_image = io.imread(image, plugin = 'pil')
        dummy_img = np.zeros(input_image.shape[0:2])

        input_blur = util.img_as_ubyte(filters.gaussian(input_image,
                                       sigma = 5,
                                       multichannel = True,
                                       truncate = 2))

        for kernel in range(1,11):
            data = image_kernel_detection.indiv_kernel_data_extraction(input_image, 
                                                                  kernel_number = kernel)

            preds = chosen_model.predict(data[['Red', 'Green', 'Blue',
                                               'Hue', 'Saturation', 'Value',
                                               'Luminosity', 'A_Axis', 'B_Axis']])

            x_data = np.array(data['X']).astype(int)
            y_data = np.array(data['Y']).astype(int)
    
            dummy_img[y_data, x_data] = preds.transpose() + 1 # Since predictions begin at 1, but not background is included, I add 1 to the predctions

            print('Kernel ' +
                  str(kernel) +
                  ' is ' +
                  str(round((sum(le.inverse_transform(preds) == 'Aleurone') / data.shape[0]) * 100, 2)) +
                  '% aleurone')

        plt.subplot(1,2,1)
        plt.imshow(input_image)
        plt.title('Original')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(dummy_img)
        plt.title('SVC (White Training)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
