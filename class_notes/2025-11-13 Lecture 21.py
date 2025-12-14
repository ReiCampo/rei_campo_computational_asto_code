
############################################################################
############################################################################
###                                                                      ###
###                           LECTURE 21 NOTES                           ###
###                                                                      ###
############################################################################
############################################################################

import numpy as np
import matplotlib.pyplot as plt
import random


##---------------------------------------------------------------
##                        Discussing GPUs                       -
##---------------------------------------------------------------

# JAX is a package for array-oriented numericaly computation. It can be used
# to replace numpy and run code much faster. It also includes a number of 
# features to improve numerical performance.

# JAX can use the GPUs, however you need to write your code in a way where it
# can be read by the GPU.

# However, there are a few differences between JAX and numpy. Arrays are 
# immutable. So once you create arrays, they can no longer be changed. Because
# the arrays are immutable, this helps speed up the process, but you lose
# flexibility in changing your arrays.

# Random numbers work differently becuase different threads need to have 
# different seeds. 

# JAX also includes just-in-time compliling. 


##----------------------------------------------------------------
##                      Parameter Estimation                     -
##----------------------------------------------------------------

# Optimization is always on the minds of people who code. However, optimization
# is actually very difficult to do. We can be general with it, but most of the
# tiem, optimization is actually difficult to work through.

# One of the first things we can work on to optimize our code is using parameter
# estimation.

# One way to estimate parameters is to use least square fitting. But when we use
# this method, what is the "best fit" line to our data? (The nomenclature sucks
# when we talk about "best fit" because it doesn't mathematically mean anything)

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