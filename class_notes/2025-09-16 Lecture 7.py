### 9/16/2025
### Lecture 7

###----- More integration -----###

# Gauss-Kronrod Quadriture
# This is nested quadrature rule, where one can perform two quadrature 
# calculations that share points.

#################################
###----- Differentiation -----###
#################################

# This is a lot easier than integration! Derivatives can always be calculated
# whereas integrals may not necessarily be calculated.

# However, there are still problems with derivatives! There is going to be 
# subtraction error and because we are dividing, division by small numbers can 
# be an issue too.

# As h gets smaller, our error gets worse. So thinking about what we should 
# choose as our h value is important.

# To determine the error, we are going to use a Taylor expansion.
# If f(x) and f''(x) are of order unity, then we should take h ~ 10e-8. This is 
# much worse because machine precision is 10e-16! 

# We can improve accuracy  by comparing forward and backward differences into a 
# central difference. If we do this, our h should be 10e-5 with an accuracy of 
# 10e-10

# This gives us a more accurate result when we calculate ther derivative not 
# necessarily at the points, but half way between the points.

#########################################
###             Excersie 1            ###
#########################################

# Create a user-defined function f(x) that returns the value 1 + 1/2tanh(2x)

import math
import matplotlib.pyplot as plt

def function(x):
    return (1 + (1/2) * math.tanh(2*x))

# Now use the central difference to calculate the derivative of the function in 
# the range [-2,2]

def central_diff(start, end, h):

    df = (function(start + (h/2)) - function(end - (h/2))) / h

    return df

derivative1 = central_diff(2, 2, 1e-5)
print(derivative1)
# Printed value: 0.001340950683825781

##############################################
###----- Higher Order Differentiation -----###
##############################################

# We can easily take second derivatives by reassigning the f'(x) to g(x) and 
# just take the first derivative again. 
# In the second derivative, we want h ~ 1e-8, and our error will be 1e-8

# We can also dertermine partial derivatives just as easily using the central 
# difference method or any other method. 

############################
###----- Noisy Data -----###
############################

# One tricky thing with derivatives is if you have noisy data. Just because you 
# can take derivatives, doesn't mean you should! 

# Think: What is the derivative actually telling me?

# Instead of taking derivatives of the noisy function, you can fit a curve to 
# the portion of the data where we want to take the derivative. This is a fit of
#  scales large enough to see the underlying function and not the noise. 

# You can also smooth the data before taking the derivative. This can be done 
# with Fourier transforms.


###############################
###----- Interpolation -----###
###############################

# Linear interpolation: Take two points closest to where you want to 
# interpolate, fit a line to them, and then from the equation of that line you 
# have the value of your previously unknown points.

# You can interpolate outside the points, if you're slightly outside the points,
# then you're okay but anything further is no good.

# Quadratic interpolation: We can use a quadratic to three points and use that 
# for the interpolation. 

# However, you can overshoot/undershoot outside of your range and can cause 
# problems if it gives nonphysical answers.

# Lagrange interpolations: This function is guaranteed to go through all the 
# points, but the function that is created isn't necessarily the correct 
# function!

# If you larger values of n for this interpolation, the end points will have 
# lots of error but not as much in the middle of your function. This may be a 
# good solution if you only care about what's happening inside the function 
# instead of at the end points.

# Splines: Interpolation but matches the derivatives of the function at the end 
# points.

# This results in a smooth appearance and avoids severe oscillations of higher 
# order polynomials, however it's really just a bunch of functions stitched 
# together, so you can't actually take derivatives. Splines are really only 
# useful for visual purposes, but not understanding the physics of the function.


################################################
###----- Linear and Nonlinear Equations -----###
################################################

# Systems of linear equations occur regularly in physics, so using a computer to
# calculate and check your work can be useful.

# One of the straightforward ways to solve a system of linear equations (SLE) is 
# to use gaussian elimination and using back substitution.

# However, this gaussian elimination is not commonly used b/c in physics we 
# solve the matrix many times. So we are looking for a method where we can solve 
# the matrix once and be done. This is where LU decomposition comes into play.

# (Look at the slides to see the example on how to do LU decomposition)

# Another thing that we are interested in are the eigenvalues/vectors. The most 
# common way to solve the eigenvalue problem is to decompose the matrix into an 
# orthogonal matrix and an upper triangular matrix.

# (See slides for further detail) If we keep multiplying, we can eventually get 
# a diagonal matrix. Each iteration of the off diagonal elements become smaller 
# and smaller.

