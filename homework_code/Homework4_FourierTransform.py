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
import argparse
import os

# Commenting this code out so that other users can run this code with their
# own directories:



def descrete_fourier_transform(input_path, N_coefficients = 10):
    '''
    This function will calculate a fourier transform and plot the power spectrum
    and fit of Fourier transform of spectra TIC 0001230647 found on the TESS 
    website: 
    https://tessebs.villanova.edu/0001230647 
    
    The function will also filter down the Julian days to 2990 to 3000, since 
    that is a region that can be fitted well by the transform.
    
    Inputs:
        input_path (str):
            Takes in the path of where your spectra data is stored.
            
        N_coefficients (int):
            The number of coefficients the user would like to use for the 
            Fourier transform. The default is set to 10 coefficients.
            
    Outpus:
        fig, ax (plot):
            Returns the inverse Fourier transform using the number of 
            coefficients inputted and animates the approximation on top of the 
            spectra.
    
    
    '''
    
    os.chdir(input_path)
    
    # Importing the chosen spectra and taking the flux and time
    tess_spectra = fits.open("tic0001230647.fits")
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
                y[n] += (coefficients[k] * exp((2j * pi * k * n) / time_step))
        
        return y / time_step
        

    # Calculating the coefficients
    dft_coefficients = coefficient_solver(flux)
    
    # Calculating the power spectrum to plot
    power_spectrum = dft_coefficients * np.conjugate(dft_coefficients)
    
    # Getting rid of k = 0 values
    plotting_power_spectrum = np.copy(power_spectrum)
    plotting_power_spectrum[0] = 0
    
    # Creating the x values for the power spectrum plot
    n = np.arange(len(plotting_power_spectrum))
    
    # Plotting the power spectrum of the Fourier series:
    plt.plot(n, plotting_power_spectrum)
    plt.title("Power Spectrum of TIC 001230647 Using " + str(N_coefficients) + " Coefficients")
    plt.xlabel("Time Steps")
    plt.ylabel("$|c|^2$")
    plt.xlim(0, 200)
    plt.show()
    
    
    # --------------------------------------------------------------------
    # The following code was taken from Claude.ai in order to fix my data used
    # when plotting my animated inverse fourier transform
    
    # Find the indices of the top N coefficients
    top_indices = np.argsort(power_spectrum[1:])[-N_coefficients:] + 1

    # Create filtered coefficients with only top N
    filtered_coefficients = np.zeros_like(dft_coefficients)
    filtered_coefficients[0] = dft_coefficients[0] 
    filtered_coefficients[top_indices] = dft_coefficients[top_indices]
    # ---------------------------------------------------------------------
    
    # Finding the flux values by calculating the inverse transform
    dft_inverse = inverse_solver(filtered_coefficients)
    
    # Making sure my x and y domains are filtered to the regoin I want to examine
    time_range = (time > 2990) & (time < 3000)
    time_plot = time[time_range]
    flux_plot = flux[time_range]
    
    # Plotting the inverse Fourier transform over the spectra
    fig, ax = plt.subplots()
    
    # Setting the limits. Currently hard-coded for now
    ax.set_xlim(2990, 3000)
    
    # Plotting the static line in the background first
    ax.plot(time_plot,
            flux_plot,
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
        animated_line.set_data(time_plot[:a], np.real(dft_inverse[:a])) # Taking the real values of dft_inverse just in case matplotlib silently fails
        return animated_line,
    
    # This is where the animation is created
    anim = animation.FuncAnimation(fig,
                                   animate_line,
                                   init_func = empty_line,
                                   frames = len(time_plot),
                                   interval = 1,
                                   blit = True,
                                   repeat = False)
    
    
    plt.title("The Inverse Transform of the Calculated Fourier Series for \n"
              "TIC 0001230647 Using " + str(N_coefficients) + " Coefficients")
    plt.xlabel("Time (Julien Days)")
    plt.ylabel("Flux")
    plt.show()
    
    return fig, ax


parser = argparse.ArgumentParser(description = "Calculate the Fourier transform " \
                                "for spectra TIC 001230647 found on the TESS " \
                                "website " \
                                "https://tessebs.villanova.edu/0001230647 using" \
                                " the number of user-inputted coefficients.")

parser.add_argument("input_path",
                    type = str,
                    help = "This is the path where your TESS data is stored. " \
                    "tic001230647.fits is hardcoded in this example, so just " \
                    "enter the path that your .fits file is located in.")

parser.add_argument("--N_coefficients",
                    default = 10,
                    type = int,
                    help = "The number of coefficients to be used when" \
                           " calculating the Fourier transform. The default is" \
                            " set to 10.")

args = parser.parse_args()

spectra_transform = descrete_fourier_transform(input_path = args.input_path,
                                               N_coefficients = args.N_coefficients)

plt.show()


