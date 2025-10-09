# 10/9/2025
# Rei Campo


############################################################################
############################################################################
###                                                                      ###
###                 CONTINUATION OF MONTE CARLO ANALYSIS                 ###
###                                                                      ###
############################################################################
############################################################################

# Before we were calculating the probability of an atom decaying, but it's much
# more useful to find the time it takes for an atom to decay. You can do this 
# by using transformation methods. 


##---------------------------------------------------------------
##                          Excercise 1                         -
##---------------------------------------------------------------

import numpy as np
import random as random
import matplotlib.pyplot as plt
rng = np.random.default_rng()


def exponential_half_life_equation(z, mu):
    x = (-1/mu) * np.log(1 - z)
    return x

particle_times = exponential_half_life_equation(rng.random(1000), 6.075)
sorted_particle_times = sorted(particle_times, reverse = True)
print(particle_times)

time_values = np.arange(1000)

# doing an initial plot to see what the array looks like
plt.plot(time_values, particle_times)
plt.xlabel("Time Steps")
plt.ylabel("Number of Particles Decayed at Time Step")
plt.title("The Number of Particles Decayed at a Given Time Step")
plt.show()

time_vals_rev = np.arange(1000, 0, -1)
# transform the plot to show how many particles are not 
plt.plot(sorted_particle_times, time_vals_rev)
plt.xlabel("Number of Particles")
plt.ylabel("Time Step")
plt.title("The Number of Particles Decayed at a Given Time Step")
plt.show()


# Choosing your random numbers from different distributions can change the way
# you solve these monte carlo analyses


##---------------------------------------------------------------
##                    Gaussian Random Numbers                   -
##---------------------------------------------------------------

# The most common probability distribution is probably the normal (or Gaussian)
# distribution. However the transformation method fails, though. However!!!!
# You can transform the equation so that you can utilize it in numerics. In this
# case, you can transform the Gaussian into polar coordinates. By doing this,
# you can now solve the transformation method for Gaussian random numbers.


##---------------------------------------------------------------
##                    Monte Carlo Integration                   -
##---------------------------------------------------------------

# Think about Rutherford scattering, the likelyhood that a particle hits a 
# nucleus is a Gaussian scattering. Meaning that the particles aren't always 
# going to hit dead center, they may hit the "top" of the nucleus, "bottom",
# etc.

# You can use Monte Carlo analysis to solve the Rutherford scattering. However!
# There is an integral that already exists. Using the Monte Carlo analysis
# produces more error. 

# You can figure out probabilities by looking at p = I / A where I = the
# integral, and A = area under the curve

# What's cool is that you don't have to actually calculate an integral! No 
# Simpson's Rule, no Trapezoidal rule. However again, this is inacurate. 
# Monte Carlo is powerful, but gives you less accuracy.

# Accuracy improves N ^ (-1/2). You will need a lot of N in order to get higher
# accuracy!

# A better way of doing Monte Carlo Integration is using the Mean Value
# Method. If we take the average value of our function in the volume, we can
# improve the accuracy a bit.


##----------------------------------------------------------------
##                          Exercise 2                           -
##----------------------------------------------------------------

N_points = 10000

def sine_func(x):
    return np.sin(1 / (x * (2 - x))) ** 2

hit_miss = 0

for i in range(N_points):
    
    x_bound = 2 * rng.random()
    
    y_bound = rng.random()
    
    if y_bound < sine_func(x_bound):
        hit_miss += 1
    
print(hit_miss)

sine_integral = 2 * hit_miss / N_points
print(sine_integral)


# Now using the mean value method:

b = 2
a = 0

avg = (b - a) / N_points

sine_mean_value_method = avg * sine_integral

print(sine_mean_value_method)


# If you are able to phrase your problem in terms of probabilities, you can use
# Monte Carlo analysis. 

# The main use of Monte Carlo Integration, though, is using it in 
# multi-dimensional space. 
