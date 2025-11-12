# 11/1/2025
# Rei Campo


#################################################################################
#################################################################################
###                                                                           ###
###  HOMEWORK 6: TAKE A BALL BEARING AND ROD FLOATING IN SPACE AND CALCULATE  ###
###    THE MOTION OF THE BALL BEARING AS IT ORBITS AROUND THE ROD. USE ODE    ###
###                TECHNIQUES TO SOLVE THE EQUATIONS OF MOTION.               ###
###                                                                           ###
#################################################################################
#################################################################################

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import argparse
import warnings

def space_debris_calculation(method, m_rod = 10, L = 2, time = 10, h_val = 0.1, x_0 = 1, y_0 = 0, vx_0 = 0, vy_0 = 1):
    '''
    This function calculates the orbit of a low mass ball (assumed massless)
    orbiting around a rood of mass m_rod and length L in space. The function 
    solves the ODE for orbits by using a 4th order Runge-Kutta or 
    Verlet-Leapfrog method. The output of the function will be a plotly graph
    that displays the orbit of the ball around the rod and its x and y positions
    with respect to time. 
    
    In this function, the universal gravitational constant G will be equal to 1,
    normalizing all other inputs like masses and velocities.
    
    Inputs:
        m_rod (float):
            The mass of the rod in normalized units. Preset to m_rod = 10
            
        L (float):
            The length of the rod that the ball is orbiting around. This length
            is also normalized. This length is automatically set to 2.
            
        time (int):
            The normalized time that the calculations is analyzed over. This 
            value is automatically set to 10.
            
        h_val (float):
            The time step you want to divide your 'time' input into. This value
            should not be larger than your 'time' input.
        
        x_0 (float):
            The starting x position of the ball. The units are normalized and is
            automatically set to 1.
            
        y_0 (float):
            The starting y position of the ball. The units are normalized and is
            automatically set to 0.
        
        vx_0 (float):
            The starting velocity of the ball in the x direction. The units are 
            normalized and is automatically set to 0.
            
        vy_0 (float):
            The starting velocity of the ball in the y direction. The units are
            normalized and is automatically set to 1.
        
        method (str):
            The input for which solver will be used to numerically calculate 
            the equations of motion of the ball. Acceptable values are:
            "RK"
            "Runge Kutta"
            "Verlet"
            
    Outputs:
        orbit_plot (plotly object):
            A plot displaying the trajectory of the ball around the rod and the
            ball's x and y positions at each time step.
    
    '''


##----------------------------------------------------------------
##                  Doing Some Error Checking:                   -
##----------------------------------------------------------------

    # Capeitalizing the user input so that it's easier to error check:
    method_adjusted = method.upper()
    
    if method_adjusted not in ("RK", "RUNGE KUTTA", "VERLET"):
        raise ValueError("The acceptable values for the method argument are: \n" \
                         "'RK', \n" \
                         "'Runge Kutta' \n" \
                         "'Verlot' \n" \
                         "Your input was: " + str(method) + " . Please check" \
                         " your spelling or use an accepted value.")
        
    if h_val > time:
        raise ValueError("The h_val you inputted was larger than your time "\
                         "value. Please input an h_val that is less than your" \
                         " time input. Your h_val input was: " + str(h_val))
    elif time / h_val < 75:
        warnings.warn("WARNING: While your h_val input is less than your" \
                        " time input, you don't have a significant amount" \
                        " of time steps. Proceeding with this h_val may not" \
                        " give you a desired amount of detail. Your current" \
                        " amount of time steps is: " + str(time / h_val))


##-------------------------------------------------------------------------------
##  Part B: Explore the orbit of the ball bearing around the rod using t = 10   -
##  and initial conditions of (x,y) = (1,0) and a velocity of +1 in the y       -
##  direction.                                                                  -
##-------------------------------------------------------------------------------

    # Setting up differential equations function:
    
    def differential_equations(x, y, x_velocity, y_velocity, Grav):
        
        r = np.sqrt(x**2 + y**2)
        
        # Defining the velocities
        # Velocity in x direction:
        u = x_velocity
        
        # Velocity in y direction:
        v = y_velocity
        
        # Defining the accelerations:
        # Acceleration in the x direction:
        dudt = -Grav * m_rod * (x / (r * np.sqrt(r**2 + (1/4) * L**2)))

        # Acceleration in the y direction:
        dvdt =  -Grav * m_rod * (y / (r * np.sqrt(r**2 + (1/4) * L**2)))
        
        # Returning an array with the calculated velocities and accelerations:   
        return np.array([u, v, dudt, dvdt])
    
    # Setting up different ODE solvers:
    
    def runge_kutta(h, x_position, y_position, x_velo, y_velo, Grav):
        
        # Calculate all four terms then add them up later:
        
        first_term = h * differential_equations(x_position, y_position, x_velo, y_velo, Grav)
        
        second_term = h * differential_equations(x_position + 0.5 * first_term[0],
                                                 y_position + 0.5 * first_term[1],
                                                 x_velo + 0.5 * first_term[2],
                                                 y_velo + 0.5 * first_term[3],
                                                 Grav)
        
        third_term = h * differential_equations(x_position + 0.5 * second_term[0],
                                                 y_position + 0.5 * second_term[1],
                                                 x_velo + 0.5 * second_term[2],
                                                 y_velo + 0.5 * second_term[3],
                                                 Grav)
        
        fourth_term = h * differential_equations(x_position + third_term[0],
                                                 y_position + third_term[1],
                                                 x_velo + third_term[2],
                                                 y_velo + third_term[3],
                                                 Grav)
        
        # Adding up the terms:
        rk_solution = (1 / 6) * (first_term + 2 * second_term + 2 * third_term + fourth_term)
        
        # Return the new positions and velocities with the respective calculated 
        # Runge-Kutta values:
        return np.array([x_position + rk_solution[0], y_position + rk_solution[1], x_velo + rk_solution[2], y_velo + rk_solution[3]])
    
    def verlet_method(h, x_position, y_position, x_velo, y_velo, Grav):
        
        # Calculating the current velocities and accelerations:
        vx, vy, ax, ay = differential_equations(x_position,
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
        
        vx_new, vy_new, ax_new, ay_new = differential_equations(new_x_position,
                                                                new_y_position,
                                                                vx,
                                                                vy,
                                                                Grav)
        
        # Finding the new velocities with the new accelerations and positions:
        new_x_velo = vx_half + (ax_new * h)
        
        new_y_velo = vy_half + (ay_new * h)
        
        return np.array([new_x_position, new_y_position, new_x_velo, new_y_velo])
    
    # Setting up initializers:
    
    grav_unit = 1
    time_steps = int(time / h_val)
    
    # Initializing arrays to later be pasted into the debris_dictionary
    
    x_pos = []
    y_pos = []
    x_vel = []
    y_vel = []
    step = []

    # Beginning to calculate the positions at each time step depending on the
    # method that the user inputted:
    
    for i in range(time_steps):
        
        # Calculate the positions of the ball at each time step using the RK 
        # method:
        if method_adjusted == "RUNGE KUTTA" or "RK":
            values_at_time_step = runge_kutta(h_val, x_0, y_0, vx_0, vy_0, Grav = grav_unit)
            
            x_pos.append(values_at_time_step[0])
            y_pos.append(values_at_time_step[1])
            x_vel.append(values_at_time_step[2])
            y_vel.append(values_at_time_step[3])
            step.append(i)
        
        # Calculate the positions of the ball at each time step using the Verlet
        # Leapfrog method:
        elif method_adjusted == "VERLET":
            values_at_time_step = verlet_method(h_val, x_0, vx_0, vy_0, grav_unit)
        
            # Updating the initialized lists that will later be put into the
            # plotting dictionary:
            x_pos.append(values_at_time_step[0])
            y_pos.append(values_at_time_step[1])
            x_vel.append(values_at_time_step[2])
            y_vel.append(values_at_time_step[3])
            step.append(i)
            
        # Updating the initial values with the newly calculated values:
        x_0, y_0 = values_at_time_step[0], values_at_time_step[1]
        vx_0, vy_0 = values_at_time_step[2], values_at_time_step[3]
        
        
    # Adding finalized lists to the debris_dictionary which will be used later
    # for plotting:

    debris_dictionary = {
        "X Position": x_pos,
        "Y Position": y_pos,
        "X Velocity": x_vel,
        "Y Velocity": y_vel,
        "Time Step": step
    }
    

##----------------------------------------------------------------
##                  Creating the Plotting Code                   -
##----------------------------------------------------------------

###  This section used Claude.ai to help with getting started with plotly. Most   
###  of the plotting code here has been suggested by Claude.ai with a couple of   
###  adjustments made to the aethetics of the plots                               

    def plotting_code(x_values, y_values, time):
        
        # Starting by making the subplots for the graph, having one row and two
        # columns 
        fig = make_subplots(rows = 1,
                            cols = 2,
                            subplot_titles = ("Orbital Path of Ball Around Rod", 
                                              "X and Y Position of Ball Orbiting Around Rod vs. Time"),
                            specs = [[{'type': 'scatter'}, {'type': 'scatter'}]], # This line ensures that both subplots are scatter plots (which are actually line plots in plotly)
                            horizontal_spacing = 0.15)
        
        # Adding in the rod to the graph in the first subplot:
        fig.add_trace(
            go.Scatter(x =[0],
                       y = [0],
                       mode = "markers", # Just selecting this mode only uses points and not lines
                       marker = dict(size = 15, color = 'red', symbol = "circle"),
                       name = "Rod, Length Pointing in Z Direction"),
            row = 1,
            col = 1)
        
        # Adding in the orbit of the ball around the rod in the first subplot:
        fig.add_trace(
            go.Scatter(x = x_values,
                       y = y_values,
                       mode = "lines+markers", # Using lines+markers means that this plot is going to have points and lines
                       marker = dict(size = 4, color = time, colorscale = "Greens"),
                       line = dict(width = 2, color = 'gray'),
                       name = "Orbit of Ball Around Rod"),
            row = 1,
            col = 1)
        
        # Adding in the x position vs time piece in the second subplot:
        fig.add_trace(
            go.Scatter(x = time, 
                       y = x_values,
                       mode = "lines", # Using lines just means that there will only be lines visible on the graph
                       name = "X Position of Ball at a Given Time Step",
                       line = dict(color = "blue")),
            row = 1,
            col = 2)
        
        # Adding in the y position vs time piece in the second subplot:
        fig.add_trace(
            go.Scatter(x = time,
                       y = y_values,
                       mode = "lines",
                       name = "Y Position of Ball at a Given Time Step",
                       line = dict(color = "red")),
            row = 1,
            col = 2)
        
        # Updates the graph according to the axes and subplot. 
        fig.update_xaxes(title_text = "X Position",
                         scaleanchor = "y",
                         scaleratio = 1,
                         row = 1,
                         col = 1)
        fig.update_yaxes(title_text = "Y Position",
                         row = 1,
                         col = 1)
        fig.update_xaxes(title_text = "Time Step",
                         row = 1,
                         col = 2)
        fig.update_yaxes(title_text = "Position",
                         row = 1,
                         col = 2)
        
        fig.update_layout(height = 500,
                          width = 1200,
                          title_text = "Orbit of a Ball Around a Rod in Space" \
                            " Using the " + str(method) + " Method",
                          title_x = 0.5,
                          title_font_size = 20)
        return fig
        
    # Plots the orbit of the ball going around the rod:
    orbit_plot = plotting_code(debris_dictionary["X Position"], 
                               debris_dictionary["Y Position"],
                               debris_dictionary["Time Step"])
    
    orbit_plot.show()
    
    return orbit_plot

parser = argparse.ArgumentParser(description = "This function calculates the" \
    " orbit trajectory of a massless ball around a rod in space. To solve" \
    " the differential equations, this function can use either a 4th Order" \
    " Runge-Kutta or Verlet solver. Acceptable values for the 'method' input" \
    " are as follows: " \
    " 'RK' " \
    " 'Runge Kutta' "
    " 'Verlet'")

# Creating the parser that takes in the method of calculation:
parser.add_argument("method", 
                    type = str, 
                    help = "This is the ordinary differential equation solver" \
                        " that the function will use to calculate the ball's "\
                        "trajectory in space. Acceptable values are:" \
                        "'RK', 'Runge Kutta', 'Verlet'")

# Creating the parser for the mass of the rod:
parser.add_argument("--m_rod",
                    default = 10, 
                    type = float,
                    help = "The mass of the rod in normalized units. The" \
                        " default is set to 10.")

# Creating the parser that takes in the length of the rod:
parser.add_argument("--L",
                    default = 2,
                    type = float,
                    help = "The length of the rod in normalized units. The" \
                        " default is set to 2")

# Creating the parser that takes in the normalized time for the calculation:
parser.add_argument("--time",
                    default = 10,
                    type = int,
                    help = "The length of time in normalized units that the " \
                        "ball orbits the rod. The default is set to 10.")

# Creating the parser that takes in the h value (aka: time step) for the 
# calculations:
parser.add_argument("--h_val",
                    default = 0.1,
                    type = float,
                    help = "The h value that will divide your time input up" \
                        " into equal time steps. The default value is set to 0.1")

# Creating the parser that takes in the initial x position of the ball:
parser.add_argument("--x_0",
                    default = 1,
                    type = float,
                    help = "The starting normalized x position of the ball." \
                        " The default value is set to 1.")

# Creating the parser that takes in the ball's initial y position:
parser.add_argument("--y_0",
                    default = 0,
                    type = float,
                    help = "The starting normalized y position of the ball." \
                        " The default value is set to 0.")

# Creating the parser that takes in the ball's initial x velocity:
parser.add_argument("--vx_0",
                    default = 0,
                    type = float,
                    help = "The initial normalized x velocity of the ball. The" \
                        " default value is set to 0.")

# Creating the parser that takes in the ball's initial y velocity:
parser.add_argument("--vy_0",
                    default = 1,
                    type = float,
                    help = "The initial normalized y velocity of the ball. The" \
                        " default value is set to 1.")


args = parser.parse_args()

orbit_trajectory = space_debris_calculation(args.method,
                                            args.m_rod,
                                            args.L,
                                            args.time,
                                            args.h_val,
                                            args.x_0,
                                            args.y_0,
                                            args.vx_0,
                                            args.vy_0)

orbit_trajectory.show()