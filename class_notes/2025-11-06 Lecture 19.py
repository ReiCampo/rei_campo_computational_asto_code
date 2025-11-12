# Rei Campo
# 11/6/2025


############################################################################
############################################################################
###                                                                      ###
###                           LECTURE 19 NOTES                           ###
###                                                                      ###
############################################################################
############################################################################


##----------------------------------------------------------------
##                      Initial Value PDEs                       -
##----------------------------------------------------------------

# Initial values problems have to look at the column of values and propagate
# forward for each column. We need a technique to do that!

# FTCS Method:
# You can start by turning the spacial dimensions into a grid of points. You can
# then take the derivative and solve your equation!
# Because you can separate the PDE into an ODE, you can use the methods from 
# ODEs as before! You can take derivatives, use Euler's methods, etc.


##----------------------------------------------------------------
##                          Exercise 1                           -
##----------------------------------------------------------------

import numpy as np

time = 365 # in days
A = 10 # in Celsius
B = 12 # in Celsius
earth_temp = 11 # in Celsius
crust_thickness = 20 # in meters
thermal_diffusivity = 0.1 # in m^2 / day
time_step = 1 # in days
epsilon = time_step / 2

resolution = 100
grid_division = crust_thickness / resolution

low_temp = -10 # in Celsius
mid_temp = 10 # in Celsius
high_temp = 35 # in Celsius

temp = np.empty(resolution + 1, float)
 

# Initial values for the project
initial_temp = 10
initial_surface_temp = 5
loop_time = 0

time1 = 30 # in days
time2 = 60 # in days
time3 = 90 # in days
time4 = 150
time5 = 300

def average_temp(t):
    temp_0 = A + B * np.sin((2 * np.pi * t) / 365)
    return temp_0


# trying for only a year first
while loop_time < len(time + 1):
    
    for i in range(1, time):
        