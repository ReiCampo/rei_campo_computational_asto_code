# 9/18/2025
# Computational Methods Lecture 8 Notes

###########################################
###         Non-Linear Equations        ###
###########################################

# There are four methods we are going to be talking about for this class:

### Method 1: Relaxation Method: 

# When given a non-linear equation, you can start by plugging in numbers and
# seeing what happens. You can iterate over and over the numbers and find the
# answer pretty quickly.
# HOWEVER!!! This will not work for all non-linear equations! For it to work, 
# you have to be able to write the function as x = f(x) (or has to be 
# rearranged in this way).
# Sometimes the function oscillates instead of relaxing to a known number.
# You can try to invert the function or rearrange the function so that it may 
# stop oscillating.

######################################
###           Exercise 1           ###
######################################

import numpy as np

def equation(x, c):
    return 1 - np.exp(-c * x)

solved_x_value = 0

i = 0

while i < 100:

    start_value = 1

    calculation = equation(start_value, 2)

    end_value = abs(calculation - start_value)

    if end_value < 1e-10:
        solved_x_value += calculation
        break
    else:
        start_value = calculation
        i = i + 1

print(solved_x_value)

# unsolved for right now


# You can use the relaxation method for equations with multiple variables


### Method 2: Bisection

# This is more robust than relaxation. It takes two numbers and plugs them into 
# the function and checks if one of the values is positive and the other is 
# negative. Then, you get closer and closer to where that "zero" point is and 
# you eventually find it.
# First you check that f(x1) and f(x2) have opposite signs, then you choose an
# accuracy epsilon
# Then you calculate the midpoint and f(midpoint)
# Repeat until you get closer and closer!
# There are shortcomings: If f(x1) and f(x2) have the same sign, you can't 
# easily find a root. This may be because there is no root between these values.
# Or, because the fucntion just touches zero but doesn't cross it!


### Method 3: Newton's Method:

# It is similar to bisection, but you use derivatives instead and use that to
# figure out to where the zero should be
# The problems with this is that the function has to be differentiable. 


######################################
###           Exercise 2           ###
######################################

def polynomial(a, b, c, d, e, f, x)
    sixth_order_poly = (a * x**6) + (-b * x**5) + (c * x**4) + (-d * x**3) + (e * x**2) + (-f * x) + 1
    return sixth_order_poly


def derivative_poly(a, b, c, d, e, f, x):
    first_derivative = (6 * a * x**5) + (5 * -b * x**4) + (4 * c * x**3) + (3 * -d * x**2) + (2 * e * x) - f
    return first_derivative

a = 924
b = 2772
c = 3150
d = 1680
e = 420
f = 42

newtons_method_value = 0

k = 0

while k < 100:

    initial_guess = 1

    deriv = derivative_poly(a, b, c, d, e, f)

# Example not solved yet

