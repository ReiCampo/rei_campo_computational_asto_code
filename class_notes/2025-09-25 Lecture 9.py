# 9/25/2025
# Lecture 9

# Reviewing over non-linear equations:

### Golden Ratio Search:

# This is a method to find a minima without derivatives. It's similar to 
# bisection If one f(x) or f(y) is less than f(a) and f(b), then we know there 
# must be a minimum between a and b. So you're essentially shrinking down until
# you reach two points that are so close to each other that in between them is 
# essentially the minima (with some allowable error).

# Your choices in points x and y should be symmetric around the center of a and 
# b.

# The downside is, though, the same as other bisection methods. You can only do 
# this with one variable, and you'll run into problems if you have multiple 
# solutions to the problem.

### Gauss-Newton Method:

# It's the same idea as using Newton's method but using higher derivatives. 

### Gradient Descent Method:

# Because it's very probable that we can't actually evaluate the second 
# derivative, we can instead use a constant (g) and take 1 / f''(x)


##### Fourier Transforms #####

# Fourier transforms allow to break down functions or signals into their 
# component parts and analyze, smooth or filter them, and gives us a way to 
# rapidly perform certain types of calculations and solve certain differential
# equations.

# Note that Fourier series can only be used for periodic functions! But if you
# have a function that is not periodic, we can replicate it over some range and
# then it will be.

# Discrete Fourier transforms are solved numerically, to some precision. In 
# math you solve for continuous Fourier transforms... But we have to solve 
# numerically with computers.

# We are going to have to calculate a lot of integrals, but that's okay! The 
# computer can handle it!

# The DFT is NOT a transform. It is exact (to machine precision). There actually
# is no approximation error for the sample of discrete amount of points. The
# error is that you aren't representing the rest of the function. 

# If your starting function is real (no imaginary numbers), we gain additional
# simplification: the C_n-r are the complex conjugates of the c_r coefficient.

# We can shift the location of the points. So far we have been going from 0 to L
# or we cdan take the midpoints of all those points.

##### Two-Dimensional Fourier Transforms #####

# We can transform multivariable functions too. Suppose we have M x N grid of 
# samples of y_mn. We take the Fourier transform and on each M row, then for
# each M row, we now have N coefficients. Then you can take the Lth coefficient
# in each M row and transform these M values again.

# Physical interpretation of these transforms break down our functions into a
# waves of different frequencies. The coefficients show us the relative
# contribution of waves at each frequency.



#######################################################
###                 Excercise 1                     ###
#######################################################

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir("/Users/RachelCampo/Desktop/CUNY Classes/Fall 2025 Computational Astro")

pandas_sunspots = pd.read_csv("Data/sunspots.txt", header = None, sep = "\t",
                              names = ["Time", "Sunspots"])

sunspot_fig, sunspot_ax = plt.subplots()

sunspot_ax.plot(pandas_sunspots["Time"], pandas_sunspots["Sunspots"],
                color = "black",
                linestyle = "-",
                linewidth = 3)

plt.show()

