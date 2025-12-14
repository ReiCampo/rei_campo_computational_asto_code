# Rei Campo
# 12/11/2025


############################################################################
############################################################################
###                                                                      ###
###                           LECTURE 27 NOTES                           ###
###                                                                      ###
############################################################################
############################################################################

# Remember, you are the one that has to analyze the results, not the computer
# The numbers that get spit out may not necessarily make any sense!

# Hyper-parameters can change the fit of the optimization process. If one person
# uses different parameters than you do, you will probably get different
# outcomes


##----------------------------------------------------------------
##                        Decision Trees                         -
##----------------------------------------------------------------

# Basically, you're playing a game of 20 questions with the machine. The idea
# is that you a question about certain features, then you keep asking questions
# until you reach the best score you can. Reference Prof. Maller's graph for a
# visual reference.

# You can use random forests to do this. Basically, you throw away some data
# as you fit the parameters. You can take multiple random forests and then take
# the average of the results. There isn't a lot we can do to figure out why the
# algorithm made the decisions it did, though.

# Gradient boosted trees starts with one signle decision tree then creates a
# second one to model the residuals from the first tree. Then it goes to a
# third, etc..

# On a different topic, you can use support vector machines. You can try to 
# classify the data by finding a line (or plane, or hyperplane) that best 
# separates the classes. 

# Neural Networks: The most popular choice of things! They are random forests
# but with weights! You can have many layers of the biases. For neural networks,
# you have to choose an activation function that converts weighted features into
# a value. This will ultimately affect the neural network functions. 
# You can engineer your features! 


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict, learning_curve
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

planet_data = sbn.load_dataset("planets")
planet_data = planet_data.drop(['method', 'number'], axis = 1)
cleaned_planet_data = planet_data.dropna()
cleaned_planet_data['Log(Mass)'] = np.log10(cleaned_planet_data["mass"])
cleaned_planet_data["Log(Distance)"] = np.log10(cleaned_planet_data["distance"])

hyper_params = {'kernel': 'rbf', 
                'C': 100,
                'gamma': 0.1,
                'epsilon': 0.1}
sv_regr = SVR(**hyper_params)

scores = cross_validate(sv_regr, 
                        cleaned_planet_data,
                        cleaned_planet_data['Log(Distance)'],
                        cv = 5, 
                        n_jobs = 5, 
                        return_train_score = True)

print(scores)
