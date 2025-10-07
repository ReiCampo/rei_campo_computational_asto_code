# 10/7/2025
# Rei Campo


############################################################################
############################################################################
###                                                                      ###
###                     MONTE CARLO ANALYSIS LECTURE                     ###
###                                                                      ###
############################################################################
############################################################################


# To get random numbers on the computer, you can use numpy.random
# default_rng(seed = insert_number_here) creates an instance of the generator 
# and rng.random(size = 2) returns 2 random floats f where 0 <= f < 1.
# rng.integers(low = 0, high = 10, size = 5) will return 5 random numbers 
# between 0 and 10
# rng.choice(a) returns a random choice from the array a

# So now we know how to generate psudorandom numbers, what do we do with them?
# Well in physics, if we know something has some probability of occurring, then 
# we create a realization of it using random numbers.

# We can think of radioactive decay in supernova. A majority of the light seen
# from the explosion comes from radiactive decay of unstable isotoptes, the 
# most important being Nickel-56 to cobalt, then iron. Let's say we have 1000
# Nickel-56 atoms, let's simulate the decay of these atoms over time, mimicking 
# the randomness of that decay using random numbers. 

# On average we know that the number N of atoms in our sample will fall off
# exponentially. 


##----------------------------------------------------------------
##                          Exercise 1                           -
##----------------------------------------------------------------

import numpy as np
import random as random
import matplotlib.pylot as plt
rng = np.random.default_rng()


number_of_ni_particles = 1000
number_of_co_particles = 0
number_of_fe_particles = 0
ni_half_life = 145.8 # this is the half life in hours
co_half_life = 1853.664 # this is the half life in hours
max_time = 2000


def half_life_equation(h, decay_time):
    hl_probability = 1 - 2**(-h / decay_time)
    return hl_probability


for i in range(number_of_fe_particles):
    
    prob_ni = half_life_equation(i, ni_half_life)
    prob_co = half_life_equation(i, co_half_life)
    
    decay = 0
    if rng.random() < prob_co:
        number_of_fe_particles += 1
    



# When you plot the above code, you are not going to get the same answer every
# time because of the different seed used to generate the random numbers.


##----------------------------------------------------------------
##                          Exercise 2                           -
##----------------------------------------------------------------

Bi_N = 10000
Po_N = 0
Ti_N = 0
Pb_N = 0

Bi_decay_time = 3633 # this is in seconds
Po_decay_time = 2.99e-7 # this is in seconds
Ti_decay_time = 183.18 # this is in seconds

Bi_values = []
Po_values = []
Ti_values = []
Pb_values = []

time_max = 20000 # in seconds
time_step = 1 # in seconds

Bi_prob = half_life_equation(time_step, Bi_decay_time)
Po_prob = half_life_equation(time_step, Po_decay_time)
Ti_prob = half_life_equation(time_step, Ti_decay_time)


for i in range(time_max):
    Bi_values.append(Bi_N)
    Po_values.append(Po_N)
    Ti_values.append(Ti_N)

    
    decay_Ti= 0
    
    for j in range(Ti_N):
        if random() < Ti_prob:
            decay_Ti += 1
    
    Ti_N -= decay_Ti
    Pb_N += decay_Ti
    
    decay_Po= 0
    
    for k in range(Po_N):
        if random() < Po_prob:
            decay_Po += 1
    
    Po_N -= decay_Po
    Pb_N += decay_Po
    
    decay_Bi = 0
    
    for l in range(Bi_N):
        prob = random()
        if prob < Bi_prob:
            decay_Bi += 1
            if prob >= 0.6406:
                Po_N += 1
            if prob <= 0.3595:
                Ti_N += 1
    
    Bi_N -= decay_Bi
    
    
