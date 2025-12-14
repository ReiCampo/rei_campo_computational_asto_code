
#################################################################################
#################################################################################
###                                                                           ###
###  CREATING A LOOSE OUTLINE FOR MY FINAL PROJECT FOR COMPUTATIONAL METHODS  ###
###                                 FALL 2025                                 ###
###                                                                           ###
#################################################################################
#################################################################################


##---------------------------------------------------------------
##        Step One: Create Classes for Trajectory Objects       -
##---------------------------------------------------------------

# First off, create classes for observing satellites and object of interest.
# Each class will hold properties that will be updated throughout the code.

class ObservingSatellite:
    
    # Create x and y positions and velocities that will update at each timestep
    x_position = [] # in code units? In AU?
    y_position = []
    x_velocity = []
    y_velocity = []
    
    # Create observing windows:
    field_of_view_x = [-1, 1] # in arcseconds?
    field_of_view_y = [-1, 1]
    
    # Adding in signal to noise ratio:
    snr = 6
    
    # Adding in how fast these satellites can turn to view the target object at
    # a later time step:
    swivel_speed = []
    
    # Have to think through more of propertiews the observing satellites will
    # have
    
class TargetObject:
    
    # Create x and y positions and velocities that will update at each timestep
    x_position = []
    y_position = []
    x_velocity = []
    y_velocity = []
    
    # Percentage of glint off of the object (aka, how much reflectivity there
    # is) from 0 to 1, where 0 is 0% reflectivity, 1 is 100% reflectivity and
    # 0.5 is 50% reflectivity:
    reflectivity = 1
    
    # Creating the shape of the target object. Will assume a smooth sphere for
    # now.
    shape = "sphere"
    
    # Again, will have to think through more properties as I go along with this
    # project
    
##----------------------------------------------------------------
##  Step Two: Create Definitions That Calculate the Darn Thing   -
##----------------------------------------------------------------

# First, create definitions for the equations of motion:

def equations_of_motion():
    velo_in_x = 0
    velo_in_y = 0
    
    accel_in_x = 0
    accel_in_y = 0
    
    return np.array([velo_in_x, velo_in_y, accel_in_x, accel_in_y])

# Next create leap frog and runge-kutta method:

def leapfrog_verlet():
    # Calculating the current velocities and accelerations:
    vx, vy, ax, ay = equations_of_motion(x_position,
                                         y_position,
                                         x_velo,
                                         y_velo,
                                         Grav)
        
    # Finding the velocity a half step forward:
    vx_half = vx + (ax * (h / 2))
        
    vy_half = vy + (ay * (h / 2))
        
    # Now finding the new positions using the half velocities from above:
    new_x_position = x_position + (h * vx_half)
        
    new_y_position = y_position + (h * vy_half)
        
    vx_new, vy_new, ax_new, ay_new = equations_of_motion(new_x_position,
                                                         new_y_position,
                                                         vx,
                                                         vy,
                                                         Grav)
        
        # Finding the new velocities with the new accelerations and positions:
    new_x_velo = vx_half + (ax_new * h)
        
    new_y_velo = vy_half + (ay_new * h)
        
    return np.array([new_x_position, new_y_position, new_x_velo, new_y_velo])

def runge_kutta():
        # Calculate all four terms then add them up later:
        
        first_term = h * equations_of_motion(x_position, y_position, x_velo, y_velo, Grav)
        
        second_term = h * equations_of_motion(x_position + 0.5 * first_term[0],
                                                 y_position + 0.5 * first_term[1],
                                                 x_velo + 0.5 * first_term[2],
                                                 y_velo + 0.5 * first_term[3],
                                                 Grav)
        
        third_term = h * equations_of_motion(x_position + 0.5 * second_term[0],
                                                 y_position + 0.5 * second_term[1],
                                                 x_velo + 0.5 * second_term[2],
                                                 y_velo + 0.5 * second_term[3],
                                                 Grav)
        
        fourth_term = h * equations_of_motion(x_position + third_term[0],
                                                 y_position + third_term[1],
                                                 x_velo + third_term[2],
                                                 y_velo + third_term[3],
                                                 Grav)
        
        # Adding up the terms:
        rk_solution = (1 / 6) * (first_term + 2 * second_term + 2 * third_term + fourth_term)
    return np.array([first_term + rk_solution[0], second_term + rk_solution[1], third_term + rk_solution[2], fourth_term + rk_solution[3]])

# Now to calculate the distance of the target object using parallax:

def calculate_parallax():
    return

# Add in errors to the prediction based on the uncertainty when calculating 
# trajectories of an object

def error_calculation():
    # This function will calculate the potential error of the satellites not 
    # getting the initial conditions of the target object correct. This function
    # will take the error and calculate all other potential trajectories that 
    # the target object may be on, then save those trajectories to be used later
    # if the satellites predicted a wrong orbit when searching for the object
    # at a later time step.
    return

# Creating the plotting function since I was thinking about potentially plotting
# many still images (basically one image for each time step) and creating a gif
# of them. Within the plotting code, I want to update the line of sight of the
# observing satellites if they do or don't see the target object

def plotting_code():
    return

##------------------------------------------------------------------------------
##  Step Three: Start setting up initial conditions and plugging things into   -
##  the functions created                                                      -
##------------------------------------------------------------------------------


##----------------------------------------------------------------
##                Step Four: Plot and Create GIF                 -
##----------------------------------------------------------------

