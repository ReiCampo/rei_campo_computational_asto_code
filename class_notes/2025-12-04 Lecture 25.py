# Rei Campo
# 12/4/2025


############################################################################
############################################################################
###                                                                      ###
###                              LECTURE 25                              ###
###                                                                      ###
############################################################################
############################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


##----------------------------------------------------------------
##                        Machine Learning                       -
##----------------------------------------------------------------

# Machine learning's goal is to 'learn' something about a data set. Learning 
# means adjusting some parameters in a model, so not what we usually think of
# as learning, but really sovling for coefficients. 

# There are a few main goals that machine learning focusses on:
# 1. Classification - Putting data into different classes (discrete)
# 2. Regression - Predicting the value of some variables based on other 
# variables (continuous)
# 3. Clustering - Combining data into groups
# 4. Dimensionality reduction - Decreasing the dimensions in your data
# 5. Anomaly detection - Finding outliers in your data

# Machine learning tasks are divided into supervised or unsupervised tasks. 
# Supervised tasks mean that we give a set of correct answers to train the
# machine. Unsupervised means that we do not provide the program with a set of
# correct answers. 

# In order to optimize the model, we need a way to evaluate when a model is 
# better. We need a way to score the model.

# To do this, we can use a confusion matrix. Depending on what you are trying to
# do, you may want to optimize the program with some combination of these things:

# Accuracy: (True Positive + True Negative) / All Tests
# Precision: True Positive / (True Positive + False Positive)
# Recall (Sensitivity): True Positive / (True Positive + False Negative)

# it's good to use different scores in your program because you can compare what
# score may be best (or if your program isn't affected by scores, which is a 
# good thing!)

# Bias vs. Variance:
# Variance is a measure of how different the results are when training the model
# on different data. Bias, however, fis the measure of how far off repeated 
# measurements are.


##---------------------------------------------------------------
##                          Example 1                           -
##---------------------------------------------------------------

planets_data = sns.load_dataset("planets")

x_training_data, y_training_data, x_test_data, y_test_data = train_test_split()