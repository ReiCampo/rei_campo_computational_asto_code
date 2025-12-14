# Rei Campo
# 11/25/2025


############################################################################
############################################################################
###                                                                      ###
###                              LECTURE 23                              ###
###                                                                      ###
############################################################################
############################################################################

import numpy as np
import matplotlib.pyplot as plt


##---------------------------------------------------------------
##                      Numerical Gravity                       -
##---------------------------------------------------------------

# Remember, gravity is the weakest of the four fundamental forces. 

# Let's start with Planetary Systems:
# Usually you have one or more massive objects (stars) and of order 10 much 
# lower mass objects.
# If you have just one star and you ignore the masses of the other planets, this 
# can be solved exactly with Kepler's Laws. 

# Adaptive time stepping is crucial to make these problems work. Remember,
# P^2 = a^3. Earth and Neptune are on a different time scale! 

# These systems are chaotic because one small change to the initial conditions
# result in exponentially divergent outcomes.

# Let's move to cosmology now:
# Since the Universe is gravity dominated due to 80% of it being made of dark
# matter. Simulating every single dark matter particle in the universe is
# impossible. You can simulate less particles and make them have larger 
# gravitational effects. 
# You have to also consider relaxation times!! What are the odds that these 
# particles going around perturb the others that are around.
# If you assume the system is collisionless, you have to account for 
# graviational softening because if 2 body encounters do happen in our system,
# we know that this isn't physical, so we remove them by using the gravitational
# softening parameter (or the smoothing length)

# Star Clusters: One of the hardes problems since you can't treat the system as
# collisionless. The other complication is that the stars themselves evolve,
# shedding mass and eventually exploding as supernova!

# Black Hole Mergers: We actually only care about two proprties: spin and mass.
# We are only interested in the gravitational waves emittedby these mergers.