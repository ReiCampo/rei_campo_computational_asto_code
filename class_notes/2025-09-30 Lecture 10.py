# 9/30/2025
# Rei Campo

# Code from the excersie from Lecture 9:

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/Users/RachelCampo/Desktop/CUNY Classes/Fall 2025 Computational Astro")

month, sun_spots = np.loadtxt('Data/sunspots.txt', unpack = True)

def dft_loop(y):
    N = len(y)
    N_real = N // 2 + 1
    count = np.arange(N)
    c = np.zeros(N_real)


### Discrete Cosine and Sine Transformations:

# We have been discussing the complex version of the Fourier series, however 
# there are advantages to just using cosine. Cosines will only fit symmetric 
# functions, but easy to construct a symmetric function by mirroring a fucntion
# and then repeating.

### Technological Applications:

# DFTs are super important and fundamental to the archetecture to how the
# internet works, and basically any way we communicate. This is becuase we want
# to store information. But how do we store that info more efficiently? We can
# take the Fourier transform and take away some variables and be able to save
# a smaller version of a file. 
# 
# This is the case for JPEGs!! A JPEG performs a 2D Fourier transform of an 
# image and stores the coefficients. It also doesn't keep some of the smaller 
# coefficients, making the file much smaller.
# 
# MPEGs do the same thing for movies, sampe for MP3s, but you are doing the DFT
# in time rather than space. The MP3 format is much more clever when choosing
# which coefficients to discard based on knowledge of what the human ear can
# and cannot hear.


### Fast Fourier Transforms:

# A DFT has to sum over N-1 values for (1/2 * N) + 1 distinct coefficients.
# This is N(1/2N + 1) ~ 1/2N^2 calculations. This is not good scaling!! We need
# a faster way of calculating these if we want to store music, videos, images,
# etc.
# 
# If we consider N = 2^m, we can break the sum of the DFT int oa sum over even
# n and a sum over odd n. You will get the same fourier transform, but at half
# the points. So in practice, one ends up needing Nlog_2(N) calculations instead
# of 1/2n^2. For a million sample points, the old way would require 5x10^11 
# calculations, however the fast way can do it in 2x10^7 calculations.
# 
# In python, these fourier transforms live in numpy.fft. The function rfft()
# will return the coefficients for a set of real sample points while fft() will
# perform the calculation for a complex set of sample points.


#####--------- Monte Carlo ---------#####

# Random Processes

# Computers are not capable of creating random numbers, but instead they can
# create pseudorandom numbers! The linear congruential random number genrator 
# is one of the most famous random number genrators.
# 
# HOWEVER! This is not random! Each value of x' following determinisistically
# from the previous value of x. If you run the program twice, it will produce
# identical results.
# 
# These results are not random enough for many uses in physics. If you are
# trying to find correlations in data, you'll find the random number correlation
# instead. This can ruin your analysis.
# 
# Instead the Mersenne twister method was pretty decent, but now it has fallen
# out of favor.  