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
import matplotlib.animation as animation
import os
import argparse

###  Setting the working directory for now to point to my local homework folder   
os.chdir("/Users/RachelCampo/Desktop/CUNY Classes/" \
"Fall 2025 Computational Astro/rei_campo_computational_asto_code/homework_code")


def descrete_fourier_transform():
    '''
    This function will calculate a fourier transform and plot the power spectrum
    and fit of fourier transform of spectra TIC 0001230647 found on the TESS 
    website: 
    https://tessebs.villanova.edu/0001230647
    
    
    '''
    
    # Importing the chosen spectra and taking the flux and time
    tess_spectra = fits.open("Homework Data/tic0001230647.fits")
    time = tess_spectra[1].data["times"]
    flux = tess_spectra[1].data["fluxes"]

    
    # Creating the function that will calculate the Fourier coefficients
    def coefficient_solver(flux_values):
        
        # Filtering down time and flux values to a desired region
        filtered_values = (time > 2990) & (time < 3000)
        
        N_x_vals = len(time[filtered_values])
        
        filtered_flux_values = flux_values[filtered_values]
        
        # Creating an empty array that can take in complex data
        coefficients = zeros(N_x_vals, complex)
        
        # Creating the for loops that will iterate over the N_x_vals and
        # calculate all combinations of Fourier coefficients.
        for k in range(N_x_vals):
            for n in range(N_x_vals):
                coefficients[k] += (filtered_flux_values[n] * exp((-2j * pi * k * n) / N_x_vals))

        return coefficients
    
    
    # Creating the function that will calculate the inverse transform so it can
    # be used later to plot against the original spectra
    def inverse_solver(coefficients):
        
        time_step = len(coefficients)
        
        y = zeros(time_step, complex)
        
        for k in range(time_step):
            for n in range(time_step):
                y[n] += (coefficients[k] * exp((-2j * pi * k * n) / time_step))
        
        return y / time_step
        

    # Calculating the coefficients
    dft_coefficients = coefficient_solver(flux)
    
    # Calculating the power spectrum to plot
    power_spectrum = dft_coefficients * np.conjugate(dft_coefficients)
    
    # Getting rid of k = 0 values
    power_spectrum[0] = 0
    
    # Sorting the list of the power spectrum to find the top 15 coefficients
    sorted_coefficients = sorted(power_spectrum, reverse = True)[0:14]
    print(sorted_coefficients)
    
    #filtering_array = sorted_coefficients < 
    
    # Creating the x values for the power spectrum plot
    n = np.arange(len(power_spectrum))
    
    # Plotting the power spectrum
    plt.plot(n, power_spectrum)
    plt.title("Plotting the Power Spectrum of TIC 0001230647")
    plt.xlabel("Time Step")
    plt.ylabel("Power Spectrum")
    plt.xlim(0, 200)
    plt.show()
    
    max_power_spec = power_spectrum.argmax()
    print(max_power_spec)
    
    dft_time_step = np.arange(len(dft_coefficients))
    print(dft_time_step)
    print(len(dft_coefficients))
    
    inverse_dft = inverse_solver(top_coefficients)
    
    time_range = (time > 2990) & (time < 3000)
    time = time[time_range]
    flux = flux[time_range]
    
    # Creating the figure
    fig, ax = plt.subplots()
    
    # Setting the limits. Currently hard-coded for now
    ax.set_xlim(2990, 3000)
    
    # Plotting the static line in the background first
    ax.plot(time,
            flux,
            label = "Spectra TIC 001230647",
            color = "#F2583A",
            linewidth = 2,
            zorder = 1)
    
    # The comma after "animated_line" turns this into a tuple which will be
    # needed later in the animation code
    animated_line, = ax.plot([], [],                               
                             label = "Inverse Transform Fit",
                             color = "#4B75DE",
                             linewidth = 2,
                             zorder = 2)
    
    ax.legend()
    
    # This is an initializer function to begins with an "empty" line
    def empty_line():
        animated_line.set_data([], [])
        return animated_line, # Again, ensuring that this remains a tuple
    
    # Creating the function that will animate the inverse transform line from
    # left to right
    def animate_line(a):
        animated_line.set_data(time[:a], inverse_dft[:a])
        return animated_line,
    
    # This is where the animation is created
    animation.FuncAnimation(fig,
                            animate_line,
                            init_func = empty_line,
                            frames = len(time),
                            interval = 0.1,
                            blit = True,
                            repeat = False)
    
    
    plt.title("The Inverse Transform of the Calculated Fourier Series for \n"
        "TIC 0001230647")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.show()
    
    return fig, ax


parser = argparse.ArgumentParser(description = "Calculate the Fourier transform " \
                                 "")
