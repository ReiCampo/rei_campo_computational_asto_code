# Rei Campo
# 12/9/2025


############################################################################
############################################################################
###                                                                      ###
###                        LECTURE 26 CLASS NOTES                        ###
###                                                                      ###
############################################################################
############################################################################

# Whenever you are working with Machine Learning algorithms, make sure you are
# always aware of the shape of the vecotrs you are using!

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict, learning_curve


planet_data = sbn.load_dataset("planets")
cleaned_planet_data = planet_data.dropna()

cleaned_planet_data['Log(Mass)'] = np.log10(cleaned_planet_data["mass"])
cleaned_planet_data["Log(Distance)"] = np.log10(cleaned_planet_data["distance"])

X, Y = cleaned_planet_data["Log(Mass)"], cleaned_planet_data["Log(Distance)"]

mass_train, mass_test, distance_train, distance_test = train_test_split(X,
                                                                        Y,
                                                                        test_size = 0.2,
                                                                        random_state = 83)