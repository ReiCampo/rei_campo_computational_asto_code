# Rei Campo
# 11/20/2025


############################################################################
############################################################################
###                                                                      ###
###                              LECTURE 22                              ###
###                                                                      ###
############################################################################
############################################################################

import numpy as np
import matplotlib.pyplot as plt

##----------------------------------------------------------------
##                  Continuation of Optimization                 -
##----------------------------------------------------------------

m = 1
b = 0
y_standard_deviation = 0.2
random_x_values = np.random.uniform(0, 1, 50)

def line(m, b, x):
    return (m * x) + b

def add_sigma(list, sigma):
    
    noisy_data = []
    
    for i in range(len(list)):
        random_noise = np.random.uniform(0, sigma)
        
        new_y_values = random_noise + list[i]
        
        noisy_data.append(new_y_values)
        
        
    return noisy_data

ideal_function = line(m, b, random_x_values)

added_noise = add_sigma(ideal_function, y_standard_deviation)

fig = plt.subplots()

plt.scatter(random_x_values, added_noise)
plt.plot(random_x_values, ideal_function, color = "red")
plt.xlabel("Random X Values Found Between 0 and 1")
plt.ylabel("Calculated Y Values")
plt.show()


# Bootstrapping:
# Subsampling from your own data. In bootstrapping, you can sample the same 
# number over and over again.


##---------------------------------------------------------------
##                          Example 1                           -
##---------------------------------------------------------------

def bootstrapping_sampling(sample_counter = 5):

    sampled_data = {}
    
    for i in sample_counter:
        random_x_sample = np.random.uniform(0, 1, 50)
        
        random_y_sample = line(m, b, random_x_sample)
        

# Jackknife resampleing: It's similar to bootstrapping however you remove one
# data point and then recalculates the estimated parameters. 

# Non-Gaussian Errors:
# The distribution of your errors will most likely not be Gaussian. So we no 
# longer have chi squared. However there are still ways you can maximize and 
# minimize your probabilities, but it is harder.

# Goodness of Fit:
# Correlated measurements      