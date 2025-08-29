### Lecutre 2 Notes
### 8/28/2025

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

#*** When you submit assignments, make sure they are only .py files so they are
#*** easily ran from the command line

#########################################
###-------- Class Excersise 4 --------###
#########################################

# Write a function that given the height of a ball determines the time it takes 
# to hit the ground. h = 1/2 g t2. Allow g to be a keyword so you can use this 
# code on other planets

# I will start by defining the function

def grav_pull(height, gravity = 9.8):
    time = ((2 * height) / gravity) ** (0.5)
    return time

# Now I will test the function by using 10 m as the initial height

# earth_time_1 = grav_pull(10)
# print(earth_time_1)
# printed message: 1.4285714285714286

#------------- Exercise 4a ------------#

# We are now going to practrice using sys.argv when running our code in the 
# command line

# When you run your python file, you will have to type a number after you write
# 'python your_code.py enter_number_here
height = float(sys.argv[1])

print(grav_pull(height), "s")

