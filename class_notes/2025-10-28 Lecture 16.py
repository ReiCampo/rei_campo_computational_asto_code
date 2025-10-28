# 10/23/2025
# Rei Campo

############################################################################
############################################################################
###                                                                      ###
###                              LECTURE 16                              ###
###                                                                      ###
############################################################################
############################################################################


##----------------------------------------------------------------
##                      Adaptive Step Size                       -
##----------------------------------------------------------------

# Sometimes you need to adjust your step sizes at certain points in the 
# function in order to get higher accuracy.

# First perform two steps of size h and one st4ep from teh same starting point 
# of size 2h.
# Next, calculate the rho ratio

# Leapfrog Method:
# Leapfrog is similar to Runge-Kutta. It has some advantages by using the 
# previous half-step to calculate the next 3/2h step
# Doing this way creates time-reversal symmetry. This is useful for when we 
# need to consider energy conservations
# Also, the error is even in step size h.
### Leapfrog may also be a good choice to use for your project!
# However, Leapfrog does not conserve energy if you do not go a full period!

# Verlet Method:
# Let's say we want to use the leapfrog method in order to solve the equations
# of motion of a system. 


##----------------------------------------------------------------
##                          Exercise 1                           -
##----------------------------------------------------------------

grav = 6.6738e-11 # in m^3 / kg * s^2

M_Sun = 1.9891e30 # in kg

d_close = 1.471e11 # in m

v_close = 3.0287e4 # in m/s

time_step = 8766 # in hours

x_positions = []

y_positions = []

x_velocity = []

y_velocity = []

for i in time_step:
    
    def orbit_position(x, y, r):
        
        x_distance = -grav * M_Sun * (x / r ** 3)
        
        y_distance = -grav * M_Sun * (y / r ** 3)
        
        return x_distance, y_distance
    
    r = orbit_position()
    
    def verlet_func(t, h):
        