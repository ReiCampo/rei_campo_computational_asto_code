### Lecutre 3 Notes
### 9/2/2025


# argparse does a lot of error checking for you. It is a separate package


#########################################
###----------- Excercise 2 -----------###
#########################################

# Generate a function with AI
# Write a function to read in a galaxy simulation in tipsy format and calculate 
# the stellar mass radial surface profile

# Starting with ChatGPT first:
# Prompt: Good morning ChatGPT, I would like you to write me a function in 
#         python that reads in galaxy simulation in tipsy format and calculates
#         the stellar mass radial surface profile.

# I am unable to import pynbody due to a wheel problem with pip, so this code
# from ChatGPT doesn't work right off the bat for me.
import pynbody
import numpy as np

def stellar_surface_density_profile(filename, nbins=50, r_max=50.0):
    """
    Reads a Tipsy simulation and computes the stellar mass surface density 
    profile.
    
    Parameters:
        filename (str): Path to the Tipsy file.
        nbins (int): Number of radial bins.
        r_max (float): Maximum radius (in kpc) for the profile.

    Returns:
        radii (ndarray): Array of bin center radii.
        surface_density (ndarray): Stellar mass surface density in each radial 
        bin (Msun/kpc^2).
    """
    # Load the simulation
    sim = pynbody.load(filename)
    
    # Convert to physical units
    sim.physical_units()
    
    # Align the galaxy so that the angular momentum vector of the stars points 
    # up
    pynbody.analysis.angmom.faceon(sim.stars)

    # Compute cylindrical radii for star particles
    stars = sim.stars
    r_cyl = np.sqrt(stars['x']**2 + stars['y']**2)
    
    # Define radial bins
    bins = np.linspace(0, r_max, nbins+1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Get masses of star particles
    masses = stars['mass']
    
    # Bin the mass by radius
    mass_in_bin, _ = np.histogram(r_cyl, bins=bins, weights=masses)
    
    # Compute area of each annular bin (pi * (R_outer^2 - R_inner^2))
    area = np.pi * (bins[1:]**2 - bins[:-1]**2)
    
    # Surface density: mass / area
    surface_density = mass_in_bin / area  # in Msun / kpc^2

    return bin_centers, surface_density
