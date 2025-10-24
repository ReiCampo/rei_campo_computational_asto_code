# 10/13/2025
# Rei Campo

####################################################################################
####################################################################################
###                                                                              ###
###      HOMEWORK EXCERCISE 5: CREATE A MONTE CARLO ANALYSIS THAT LOOKS AT       ###
###  SCATTERING EVENTS IN BOTH A 1 KM SLAB WITH UNIFORM ELECTRON DENSITY AND IN  ###
###                                    THE SUN                                   ###
###                                                                              ###
####################################################################################
####################################################################################


##----------------------------------------------------------------
##                  Importing necessary packages                 -
##----------------------------------------------------------------

import numpy as np
import random as random
import matplotlib.pyplot as plt
rng = np.random.default_rng()
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import argparse


##----------------------------------------------------------------
##            Setting up Monte Carlo Analysis function           -
##----------------------------------------------------------------

def photon_mean_free_path(N_photons, scattering_arena):
    '''
    This function will calculate the mean free path of photons in a given
    density using a Monte Carlo analysis. The user can either look at photons
    scattering in a uniform slab of one kilometer or look at the path of the
    photons after 1 million scatters in the Sun.
    
    Inputs:
        N_photons (int):
            The number of photons that will scatter. All photons will start in
            the center of the area of interest. The number of photons a user
            could go up to 20.
        
        scattering_arena (str):
            The area of interest that the photons will scatter through. 
            Acceptable inputs are either "Slab" or "Sun".
    
    '''
    
##----------------------------------------------------------------
##                        Error Checking                         -
##----------------------------------------------------------------

    if N_photons > 20:
        raise ValueError("You have selected too many photons. (User input: " +
                         str(N_photons) + ") Please choose a number less than" \
                         " 20")
        
    scattering_arena = scattering_arena.upper()
    
    
##---------------------------------------------------------------
##          Declaring Global Variables for the Function         -
##---------------------------------------------------------------


    # The Thompson cross section of a photon hitting an election
    thomspson_cross_section = 6.652e-25 # The units are in cm^2
    
    # This dictionary will be used later to store the X and Y positions of the
    # photon as it scatters.
    photon_path_position = {}
    
    # Creating a random color generator so each photon has their own unique
    # color when animating:
    def random_colors():
        color = random.choice(list(mcolors.CSS4_COLORS.keys()))
        return color
    
    
##---------------------------------------------------------------
##            Monte Carlo Analysis for Uniform Slab             -
##---------------------------------------------------------------


    if scattering_arena == "SLAB":
        
        electron_density = 1e20
        
    # This is the average distance the photon goes before a scattering event
        mean_free_path = 1 / (electron_density * thomspson_cross_section)
    
        slab_length = 1e5 # This is in cm
    
        slab_width = 1e5 # This is in cm
    
        # The particle will start at the bottom middle edge of the slab of electons
        # and go on its path until it hits into an electron and deflects elsewhere. 
        # Because we are considering an isotropic example, all directions are 
        # equally likely. I will only use a 2-D space for now
    
        for i in range(N_photons):
        
            x_position = [0.0] # In cm
        
            y_position = [0.0] # In cm
        
            # Setting up initializers to use in the while loop
            current_x = 0.0 # in cm
        
            current_y = 0.0 # in cm
        
        # Setting up a random angle that the photon is going to be traveling in
        # using from a range of 0 to pi initially.
            theta = random.uniform(0, 2 * np.pi)
            
            while (-slab_width / 2 <= current_x <= slab_width / 2 and
                    -slab_length / 2 <= current_y <= slab_length / 2):
            
                # Finding the distance the photon moved in the radial directeion 
                # based on the distribution equation (we don't know which direction 
                # yet b/c the angle has not been used yet):
                random_number = random.uniform(0, 1)
                distance_distribution = (-mean_free_path) * np.log(1 - random_number)
            
                # Calculating the updated x and y position of the photon:
                current_x = current_x + (distance_distribution * np.cos(theta))
            
                current_y = current_y + (distance_distribution * np.sin(theta))
            
                # Appending that new position to the initialized list:
                x_position.append(current_x)

                y_position.append(current_y)
            
            
                # Checking to see if the particle is still in the slab. If not, 
                # break from the loop:
                if not (-slab_width / 2 <= current_x <= slab_width / 2 and
                        -slab_length / 2 <= current_y <= slab_length / 2):
                    break
            
                # Randomly find the new theta angle that the photon is moving after
                # the scatter event:
                theta = random.uniform(0, 2 * np.pi)
                
            # Updating the dictionary list with the photon's name:
            photon_name = "Photon " + str(i)
                
            photon_path_position[photon_name] = {
                "X Position (in cm)": x_position,
                "Y Position (in cm)": y_position
            }
        
        
##----------------------------------------------------------------
##                Plotting Result for Uniform Slab               -
##----------------------------------------------------------------

###  There are various sections in this plotting code where I used ClaudeAI to   
###  help me, however I have adjusted the AI code to suit my purposes of this    
###  homework, so there isn't a place where I can directly point out what came   
###  from AI, so I will just make a statement here that this section used AI     
###  code.                                                                       

        # Creating the animated plot  
        
        fig, ax = plt.subplots()
    
        # Creating a transparent slab:
        ax.fill_between(x = [-1e5, 1e5], 
                        y1 = -1e5,
                        y2 = 1e5,
                        color = "#D4A419",
                        alpha = 0.2,
                        zorder = 2,
                        label = "Uniform Electron Slab")
    
        # Creating initializers so I can call them later when animating the plot:
        line_info = []
        data_info = []
        text_info = []
    
    
        for photon, positions in photon_path_position.items():
        
            # Pulling out the x and y values for each photon:
            x_vals = positions["X Position (in cm)"]
            y_vals = positions["Y Position (in cm)"]
        
            # Appending the x and y values to the initalizers above:
            data_info.append((x_vals, y_vals))
        
            # Creating the photon line. The comma at the end of photon_line 
            # ensures that photon_line remains a tuple which is needed later
            # when creating the animation.
            photon_line, = ax.plot([], [],
                                    label = photon,
                                    color = random_colors(), # Select random color for this photon
                                    marker = "o",
                                    markersize = 5)
        
            line_info.append(photon_line)
            
            # Adding a text label for each photon so the name can follow the 
            # photon's path:
            photon_text = ax.text(0, 0,
                                  photon,
                                  fontsize = 8,
                                  ha = "left",
                                  va = "bottom")
            
            text_info.append(photon_text)
    
        ax.set_title(str(N_photons) + " Photons Scattering Through a Uniformly Dense \n"
                    "Electron Slab of 1 Kilometer")
        ax.set_xlabel("Distance (In cm)")
        ax.set_ylabel("Distance (In cm)")
        ax.set_xlim(-2e5, 2e5)
        ax.set_ylim(-2e5, 2e5)
        ax.legend(loc = "lower right")
        
        # Creating the empty line function:
        def empty_line():
            for line in line_info:
                line.set_data([], [])
                
            # Adding in the empty text and position to ensure that the text
            # follows the point as it animates:
            for text in text_info:
                text.set_position((0, 0))
                text.set_text('')
            return line_info + text_info
        
        # Creating the animated line. This will animate the photon's path and 
        # the corresponding photon's text.
        def animate_line(a):
            for i in range(len(line_info)):
                line = line_info[i]
                text = text_info[i]
                x_vals, y_vals = data_info[i]
            
                end_index = min(a + 1, len(x_vals))
                line.set_data(x_vals[:end_index], y_vals[:end_index])
                
                if end_index > 0:
                    text.set_position((x_vals[end_index - 1], y_vals[end_index - 1]))
                    text.set_text(f"Photon {i}")
            return line_info + text_info
        
        # This ensures that the animation goes to the maximum amount of frames
        # in case one photon stops scattering before another.
        max_frames = max(len(data[0]) for data in data_info)
        
        anim = animation.FuncAnimation(fig,
                                        animate_line,
                                        init_func = empty_line,
                                        frames = max_frames,
                                        interval = 100,
                                        blit = True,
                                        repeat = False)
    
##----------------------------------------------------------------
##                  Monte Carlo Analysis for Sun                 -
##----------------------------------------------------------------

 
    elif scattering_arena == "SUN":
        
        # Radius of the Sun:
        R_sun = 6.9634e6 # in cm
         
        # Creating the function to calculate the electron density given the
        # radius of the Sun:
        def sun_electron_density(r):
            ne_r = 2.5e26 * np.exp(-r / (0.096 * R_sun)) #in cm^-3
            return ne_r
        
        for i in range(N_photons):
            # Initializing a list to be appended to for later
            x_distance_from_center = [] # in cm
            
            y_distance_from_center = [] # in cm
            
            current_x_position = 0 # in cm
            
            current_y_position = 0 # in cm
            
            # Running the simulation code out to 1 million scatters. This is 
            # because if I ran it out to the point where the photon leaves the
            # Sun, this code would take a long time to run.
            while(len(x_distance_from_center) <= 1e6 and len(y_distance_from_center) <= 1e6):
                
                # Calculating the radial distance from the Sun:
                radial_distance = np.sqrt(current_x_position**2 + current_y_position**2)
                
                # Calculating the electron density at that radial distance:
                density_at_distance = sun_electron_density(radial_distance)
                
                # Recalculating the mean free path with the new density:
                current_mean_free_path = 1 / (density_at_distance * thomspson_cross_section)
                
                
                random_probability = random.uniform(0, 1)
                random_distance = (-1 / current_mean_free_path) * np.log(1 - random_probability)
                
                # Finding the random scattering angle at the beginning:
                random_angle = random.uniform(0, 2 * np.pi)
                
                # Calculating the updated x and y position of the photon:
                current_x_position = current_x_position + (random_distance * np.cos(random_angle))
            
                current_y_position = current_y_position + (random_distance * np.sin(random_angle))
            
                # Appending that new position to the initialized list:
                x_distance_from_center.append(current_x_position)

                y_distance_from_center.append(current_y_position)
                
            
            # Because there are so many scattering events, I am going to reduce
            # these lists down so the animation later will be more interesting
            # to look at:
            
            # I want to keep the end points, though, so I'm setting up some 
            # coditionals first:
            if len(x_distance_from_center) > 0:
                selected_x_distance = [x_distance_from_center[0]] + x_distance_from_center[10000::10000]
                selected_y_distance = [y_distance_from_center[0]] + y_distance_from_center[10000::10000]
                
            # Appendoing the last scattering event by checking if it the length
            # of the x_distance_from_center list has a remainder. If it has a 
            # remainder, that means the slicing above won't include the last 
            # endpoint:
            if len(x_distance_from_center) % 10000 != 0:
                selected_x_distance = selected_x_distance + [selected_x_distance[-1]]
                selected_y_distance = selected_y_distance + [selected_y_distance[-1]]
                
            # Updating the dictionary list with the photon's name:
            photon_id = "Photon " + str(i)
                
            photon_path_position[photon_id] = {
                "X Position (in cm)": selected_x_distance,
                "Y Position (in cm)": selected_y_distance
            }
        
      
##----------------------------------------------------------------
##                    Plotting Results For Sun                   -
##----------------------------------------------------------------

###  There are various sections in this plotting code where I used ClaudeAI to   
###  help me, however I have adjusted the AI code to suit my purposes of this    
###  homework, so there isn't a place where I can directly point out what came   
###  from AI, so I will just make a statement here that this section used AI     
###  code.                                                                       
          
        # Creating the animated plot  
        
        fig, ax = plt.subplots()
        
        # Creating a transparent cirlce to represent 0.05 * the radius of the
        # Sun:
        sun_area = plt.Circle((0, 0),
                              radius = 0.05 * R_sun,
                              color = "#D4A419",
                              alpha = 0.2,
                              label = "0.05 of Sun's Radius")
    
        ax.add_patch(sun_area)
    
        # Creating initializers so I can call them later when animating the plot:
        line_info = []
        data_info = []
        text_info = []
    
    
        for photon, positions in photon_path_position.items():
        
            # Pulling out the x and y values for each photon:
            x_vals = positions["X Position (in cm)"]
            y_vals = positions["Y Position (in cm)"]
        
            # Appending the x and y values to the initalizers above:
            data_info.append((x_vals, y_vals))
        
            photon_line, = ax.plot([], [],
                                    label = photon,
                                    color = random_colors(),
                                    marker = "o",
                                    markersize = 2)
        
            line_info.append(photon_line)
            
            # Adding a text label for each photon so the name can follow the 
            # photon's path:
            photon_text = ax.text(0, 0,
                                  photon,
                                  fontsize = 8,
                                  ha = "left",
                                  va = "bottom")
            
            text_info.append(photon_text)
    
        ax.set_title(str(N_photons) + " Photons Scattering Through the Sun")
        ax.set_xlabel("Distance (In cm)")
        ax.set_ylabel("Distance (In cm)")
        ax.set_xlim()
        ax.legend(loc = "lower right")
        
        def empty_line():
            for line in line_info:
                line.set_data([], [])
            
            for text in text_info:
                text.set_position((0, 0))
                text.set_text('')
            return line_info + text_info
        
        def animate_line(a):
            for i in range(len(line_info)):
                line = line_info[i]
                text = text_info[i]
                x_vals, y_vals = data_info[i]
            
                end_index = min(a + 1, len(x_vals))
                line.set_data(x_vals[:end_index], y_vals[:end_index])
                
                if end_index > 0:
                    text.set_position((x_vals[end_index - 1], y_vals[end_index - 1]))
                    text.set_text(f"Photon {i}")
            return line_info + text_info
        
        max_frames = max(len(data[0]) for data in data_info)
        
        anim = animation.FuncAnimation(fig,
                                        animate_line,
                                        init_func = empty_line,
                                        frames = max_frames,
                                        interval = 75,
                                        blit = True,
                                        repeat = False)      
    
    return anim


parser = argparse.ArgumentParser(description = "This function uses a Monte Carlo analysis to" \
    " calculate the scattering events of a photon off of electrons in a 1 km" \
    " slab or within the Sun.")

# Creating the parser that takes the number of desired photons.
parser.add_argument("N_photons", 
                    type = int, 
                    help = "These are the number of photons you would like to" \
                        " track in the area of interest. The maximum amount of" \
                        " photons a user can input is 20.")

# Creating the parser that takes in the user's area of interest.
parser.add_argument("scattering_arena", 
                    type = str,
                    help = "This is the area of interest the user would like to" \
                        " perform the Monte Carlo analysis. Acceptable answers" \
                        " are either 'Slab' or 'Sun'.")


args = parser.parse_args()

mc_analysis = photon_mean_free_path(args.N_photons,
                                    args.scattering_arena)

plt.show()