# 10/16/2025
# Rei Campo

############################################################################
############################################################################
###                                                                      ###
###                              LECTURE 12                              ###
###                                                                      ###
############################################################################
############################################################################


##---------------------------------------------------------------
##              Continuation of Monte Carlo Lecture             -
##---------------------------------------------------------------

import numpy as np
import random as random
import matplotlib.pyplot as plt
rng = np.random.default_rng()

# Notes for homework: 
# The mean free path is the probability we are checking against, not the 
# Thompson scattering!!!

# n_e is the probability we are checking against in the sun. When does the 
# photon scatter?

# This calculation is going to be intensive! Make sure to start with only one or
# two photons, then increase after you see that your code is working okay. 

# Time your code!!! This is called profiling. You can use this code:

import time

start = time.time()

# code code code code code code #

end = time.time()

print(f"Program took :{end - start} seconds to run")

# You can also use a package called "timeit" and calculate the time it takes 
# for a specific function to run. It takes the average over a million times ran

# tqdm gives you a status bar as it goes through a loop.


##---------------------------------------------------------------
##                        Lecture Notes                         -
##---------------------------------------------------------------

# Monte Carlo shines in high dimensionality. Integration in many dimension can
# use mean value theorems.

# You can always transform your random numbers into another fucntion. This can
# help when you are weighting your points since some points are more "important"
# than others. 

# You don't always have to use your first probabilities in MC, you can shift 
# the probabilities that can actually lead to more accuracy!

##---------------------------------------------------------------
##                          Excersise 1                         -
##---------------------------------------------------------------

N_points = 10000000
inside = 0

for i in range(N_points, n_dimensions):
    
    x_direction = random.uniform(-1, 1)
    y_direction = random.uniform(-1, 1)
    
    if x_direction**2 + y_direction**2 <= 1:
        inside += 1

circle_area = 4 * (inside / N_points)    