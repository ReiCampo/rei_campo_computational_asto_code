# 10/23/2025
# Rei Campo

############################################################################
############################################################################
###                                                                      ###
###                        LECTURE 15 CLASS NOTES                        ###
###                                                                      ###
############################################################################
############################################################################


##----------------------------------------------------------------
##                      Continuation of ODEs                     -
##----------------------------------------------------------------

# As we look over the slides of the lecture, f(x,t) gives you the derivative,
# just remember that as you look back over the slides!

# First thing with ODEs, you need to start with the boundary condition.
# We can use Euler's method by taking a Taylor Expansion in order to find the 
# boundary condtion. However, we can get higher accuracy using Runge-Kutta's
# method (and it's faster!).


##----------------------------------------------------------------
##                          Exercise 1                           -
##----------------------------------------------------------------

import numpy as np
import scipy.constants as const

pressure_profile = 101325 # this is in N/m^2

temp = 300 # this is in K

mu = 29

k_b = const.Boltzmann

proton_mass = const.m_p

distance_range = np.arange(0, 40e3) # this is in meters

gravity = 9.81 # this is in m/s

h = 1 / len(distance_range)

def pressure_func(pressure):
    new_pressure = ((-mu * proton_mass) / (k_b * temp)) * pressure
    return new_pressure


for i in range(len(distance_range)):
    
    solved_pressure = pressure_func(pressure_profile) + h * pressure_func(pressure_profile)
    
    pressure_profile += solved_pressure


# Runge-Kutta takes the slope (derivative) but splits it in half instead. This
# is why it's mroe accurate! Think of it as going from trapezoidal rule to 
# Simpson's rule!

# Careful, make sure you've coded your Runge-Kutta correctly. It's very 
# accurate, so even when you get one term wrong, you won't notice the 
# difference! 

# What if you have more than one variable? Well, it's not much more difficult 
# than what we've been doing before!

# Also, we usually have second order derivatives in physics, so first order 
# derivatives aren't very useful. 

def f(r, t):
    theta = r[0]
    omega = r[1]
    
    ftheta = omega
    fomega = -(g/l)*np.sin(theta)
    
    
##----------------------------------------------------------------
##                          Exercise 2                           -
##----------------------------------------------------------------

pend_length = 0.1 # this is in m

start_angle = 179 # this is in degrees

g = -9.81 # this is in m/s^2



def pendulum_function(r):
    theta = r[0]
    omega = r[1]
    
    theta_diff = omega
    omega_diff = -(g / pend_length) * np.sin(theta)
    
    return np.array([theta_diff, omega_diff], float)
    
def rungekutta():
    
    