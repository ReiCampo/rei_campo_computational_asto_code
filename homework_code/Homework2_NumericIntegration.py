# 9/12/2025
# Rei Campo
# Computational Methods in Astrophysics Homework 2

### ------------------------------------------------------------------------ ###
###                           Homework Excercise 2:
###  Given E(x) = integral(0, x)[e ^ (-t^2) dt], write a program to calculate
###  E(x) for the values of x from 0 to 3 in steps of 0.1. Choose whatever
###  method you would like to calculate this integral with a suitable amount
###  of slices. Then, once your program is working, create a graph of E(x) as
###                             a function of x. 
### ------------------------------------------------------------------------ ###

### The following comment style is used to denote where I used AI to help code 
### this program:  #***#
### I've created this symbol so it is easier to "ctrl+f" and find the AI lines.


# Importing necessary packages:

import numpy as np
import matplotlib.pyplot as plt
import argparse

# Beginning of function that will calculate the integral and plot it in the 
# xy plane:

def numerical_integration(start, end, steps, method):
    '''
    This function will calculate the integral of e^(-t^2) over a user-inputted
    interval and steps. The function will calculate the integral with a 
    Trapezoidal Rule or a Simpson's rule, while also calculating the 
    error of each output. Additionally, a graph will be created to show the
    function's curve.

    Inputs:
        start (float):
            This is the number of the start of the interval that the user wants
            to integrate over.

        end (float):
            This is the number at the end of the interval that the user wants
            to integrate over.
        
        steps (float):
            This is the number of steps the user would like to use in between
            the 'start' and 'end' interval.

        method (str):
            The integration method that the user wishes to calculate. Acceptable
            values are:
            "Trapzoidal",
            "Simpsons"

    
    Outputs:
        fix, ax (plot):
            A plot displaying the points that were calculated in the number of
            steps provided. This graph will display what method was used, the 
            interval, and step sizes.

    
    '''

    # Creating some initial error checks for the function inputs. The 'start',
    # 'end', and 'steps' arguments must be a float or int while the 'method'
    # argument must be a string. If any of these arguments are not the type
    # they should be, the following errors will be thrown:

    if isinstance(start, float | int) == False:
        raise ValueError("The 'start' argument requires you to input a number" \
        "(i.e. a float or int). You inputted: " + str(start))
    
    elif isinstance(end, float | int) == False:
        raise ValueError("The 'end' argument requires you to input a number" \
        "(i.e. a float or int). You inputted: " + str(end))
    
    elif isinstance(steps, float | int) == False:
        raise ValueError("The 'steps' argument requires you to input a number" \
        "(i.e. a float or int). You inputted: " + str(steps))
    
    
    # The code will now do more error checking for the 'method' argument. It 
    # will first capitalize the entire string to ensure that there are no 
    # string mismatches. For example, the code will not be case sensitive if 
    # a user inputs method = "trapezoidal" or method = "Trapezoidal". Once the
    # string is capitalized, it will then check to see if the inputted string is
    # an acceptable value.

    # Creating error check for method:

    if isinstance(method, str) == True:

        # Capitalizing the string the user inputted:

        method = method.upper()

        if method not in ('TRAPEZOIDAL', 'SIMPSONS'): #***# The 'not in ()' code I got from ChatGPT 
            raise ValueError("The acceptable integration methods are the " \
            "following: \n" \
            "Trapezoidal\n" \
            "Simpsons\n" \
            "You inputted: " + str(method) + ".  Please use one of the "
            "following methods listed above or check your spelling.")
            
    elif isinstance(method, str) == False:
        raise ValueError("The 'method' argument requires you to input a string." \
        "You inputted: " + str(method) + ".  Please use 'Trapezoidal' or "\
        "'Simpsons' as your 'method' input.")
            


    # Creating the exponential function that will be numerically integrated over
    # later:

    def exponential_function(x):
        return(np.exp(-x**2))

    # Starting with initializing the variable that will hold the values of the
    # integral at each step:

    integral_calculation = []

    # Now calculating delta_x and making it a local variable to this function:

    delta_x = (end - start) / steps

    # Now creating a range of values that will later be my x values for plotting
    # and calculating the numerical integrals.

    x_values = np.arange(start, end, delta_x)

    # Calculating the integral for the exponential_function depending on the 
    # user-inputted method:

    if method == "TRAPEZOIDAL":

        # Because I want to calculate the area of each trapezoid at each step, 
        # I'm going to have to create nested for loops. The first for loop will 
        # iterate over my x_value range (aka, the steps between my start and end 
        # points) and then it will iterate over the summation calculation found 
        # in the equation for trapezoid rule.

        for i in x_values:

            # Calculating the delta_x at each step so I can find the area under
            # that trapezoid later:

            delta_x_end_point = (i - start) / steps
        
            # Calculating the front terms to be used later:

            front_terms = 0.5 * exponential_function(start) + 0.5 * exponential_function(i)

            # Now setting up an initalizing value for calculating the area under 
            # the trapezoid for later:

            trapezoid_area = 0

            # Now calculating the summation by creating a for loop to 
            # numerically integrate at each step. 

            for x in range(0, int(steps)):
                trapezoid_area += exponential_function(start + (x * delta_x_end_point))

            # Appending the summed values to the empty list initialized earlier

            integral_calculation.append(delta_x_end_point * (front_terms + trapezoid_area))

    
    elif method == "SIMPSONS":

        # Starting off by making sure that the 'steps' are an even number so
        # Simpson's Rule can be calculated properly:

        if steps % 2 != 0:
            raise ValueError("Because you are using Simpson's method, you need " \
            "to use an even step size. \n " \
            "Your input was: " + str(steps) + ". \n"
            "Please use a different step size if you want to continue to use " \
            "Simpson's method.")
        
        # Now moving on to the actual calculation. Again, this is very similar
        # to the trapezoidal rule above. I want to calculate the area of each
        # parabola under the curve. I will first go through each x_value (aka: 
        # the step size) and calculate delta_x for those end points, then I 
        # will calculate the summations found for Simpson's Rule.

        for i in x_values:

            # Starting by calculating the delta_x for the current endpoint the 
            # code is at:

            delta_x_at_end_point = (i - start) / steps

            # Setting initializers for later when the code calculates the area 
            # under the parabola:

            first_sum_area = 0

            second_sum_area = 0

            # Calculating the front two terms at this end point:

            front_terms = exponential_function(start) + exponential_function(i)

            # Now calculating the summation terms:

            for j in range(0, int(steps / 2) + 1):
                first_sum_area += 4 * exponential_function(start + (2*j - 1) * delta_x_at_end_point)

            for k in range(0, int(steps / 2)):
                second_sum_area += 2 * exponential_function(start + 2 * k * delta_x_at_end_point)

            # Appending to the earlier initialized list so I can later use these
            # as my y values for when I plot:
            
            integral_calculation.append((delta_x_at_end_point / 3) * (front_terms + first_sum_area + second_sum_area))

    # Now that the numerical integration is complete, the code will plot the y 
    # values for every x_value (remember, 'x_value' is the step size between the 
    # user-inputted 'start' and 'end' values)

    fig, ax = plt.subplots()
    
    # First creating a line plot that will be in the background:

    ax.plot(x_values,
             integral_calculation,
             color = "black",
             linestyle = "-",
             linewidth = 3,
             zorder = 1) #***# I got this "zorder" line of code from ChatGPT
    
    # Next, plot the calculated points over the line plot:

    ax.scatter(x_values, 
                integral_calculation, 
                color = "#C159F0",
                marker = "D",
                s = 100,
                edgecolors = "black",
                linewidths = 3,
                label = "Calculated values using the \n " + str(method) + " method",
                zorder = 2) #***# I got this "zorder" line of code from ChatGPT
    
    # Then, plot the shaded region under the graph:

    ax.fill_between(x_values,
                    integral_calculation,
                    color = "gray",
                    alpha = 0.5,
                    label = "Area under the curve using \n the " + str(method) + " method",
                    zorder = 3) #***# I got this "zorder" line of code from ChatGPT
    
    # Setting the title and labels for the graph:

    ax.set_title("Calculating the Numerical Integral Using the " + str(method) + " Method")
    ax.set_xlabel("Steps Calculated From " + str(start) + " to " + str(end))
    ax.set_ylabel("Numerically Integrated Value")
    ax.legend(loc = "lower right",
              fontsize = 8,
              markerscale = 0.8)
    ax.grid()


    return fig, ax

# Creating parser so the code can be called on the command line:

parser = argparse.ArgumentParser(description = "Numerically integrate e^(-x^2) using either Trapezoid or Simpson's Rule and plot the results")

# Creating the start parser
parser.add_argument("start", 
                    type = float, 
                    help = "This is the beginning of the interval you would like " \
                    "to numerically integrate over.")

# Creating the end parser
parser.add_argument("end", 
                    type = float,
                    help = "This is the end of the interval you would like to " \
                    "numerically integrate over.")

# Creating the steps parser
parser.add_argument("steps",
                    type = float,
                    help = "The number of steps you would like break your " \
                    "interval up into. For example, if your 'start' and 'end' " \
                    "inputs are 0 and 5 respectively, a 'steps' input of 10 " \
                    "would break the interval up into 10 pieces:" \
                    "[0, 0.5, 1, 1.5, 2, 2.5, 3, 3,5, 4, 4.5]. If you are using " \
                    "Simpson's rule, however, your steps MUST be an even number.")

# Creating the method parser
parser.add_argument("method",
                    type = str,
                    help = "The numerical integration method you would like to " \
                    "use. This function can take either Trapezoidal or Simpson's " \
                    "Rule. To select a method, please write either 'Trapezoidal' " \
                    "or 'Simpsons'.")

args = parser.parse_args()

numerical_plot = numerical_integration(args.start,
                                       args.end,
                                       args.steps,
                                       args.method)

plt.show()


