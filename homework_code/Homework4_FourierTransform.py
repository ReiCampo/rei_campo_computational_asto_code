# 10/3/2025
# Rei Campo
# Computational Methods in Astrophysics Homework 4


#################################################################################
#################################################################################
###                                                                           ###
###  HOMEWORK EXCERCISE 4:  CHOOSE A FITS FILE WITH SPECTRA DATA FROM TESS.   ###
###  THEN, CREATE A FOURIER TRANSFORM FOR A SPECIFIC EPOCH OF THE DATA, THEN  ###
###            PLOT THE POWER SPECTRUM WITH THAT FOURIER TRANSFORM            ###
###                                                                           ###
#################################################################################
#################################################################################


##---------------------------------------------------------------
##                Importing necessary packages:                 -
##---------------------------------------------------------------

from astropy.io import fits
import numpy as np
from numpy import zeros
from cmath import exp, pi
import matplotlib.pyplot as plt
import os

###  Setting the working directory for now to point to my local homework folder   
os.chdir("/Users/RachelCampo/Desktop/CUNY Classes/" \
"Fall 2025 Computational Astro/rei_campo_computational_asto_code/homework_code")


##--------------------------------------------------------------------------
##  Creating some tests to see where I want to start my Fourier analysis   -
##--------------------------------------------------------------------------

###  Using spectra data from website https://tessebs.villanova.edu/0001045298:   
###  TIC 1045298                                                                 
tess_spectra = fits.open("Homework Data/tic0001045298.fits")
time = tess_spectra[1].data["times"]
flux = tess_spectra[1].data["fluxes"]


###  Plotting the flux vs time graph just to see what the spectra looks like:   
plt.plot(time, flux)
plt.xlabel("Time")
plt.ylabel("Flux")
plt.xlim(1426, 1430) # Chose these x limits just to get a few cycles
plt.show()


###  Trying to create a fourier transform:                       
### These solve for the coefficients of the fourier transform
def discrete_fourier_transform(y_values):
    N = len(y_values)
    c = zeros((N // 2) + 1, complex)
    
    for k in range((N // 2) + 1):
        for n in range(N):
            c[k] += (y_values[n] * exp((-2j * pi * k * n) / N))
        
        return c
    
power_spectrum = discrete_fourier_transform(flux) * 
    
test_transform_1 = discrete_fourier_transform(flux)
print(test_transform_1)


def descrete_fourier_transform():
    '''
    This function will calculate a fourier transform and plot the power spectrum
    of spectra TIC 1045298 found on the TESS website 
    https://tessebs.villanova.edu/0001045298

    Inputs:
        flux_values:
            These are the flux values inputted from
    
    '''
    
    tess_spectra = fits.open("Homework Data/tic0001045298.fits")
    time = tess_spectra[1].data["times"]
    flux = tess_spectra[1].data["fluxes"]

    
    def coefficient_solver(flux_values):
        
        N_xvals = len(range(1426, 1430))
        
        c = zeros((N_xvals // 2) + 1, complex)
        
        for k in range((N_xvals // 2) + 1):
            for n in range(N_xvals):
                c[k] += (flux_values[n] * exp((-2j * pi * k * n) / N_xvals))
                
        return c
    
    
        
        
        
        
        
    
    