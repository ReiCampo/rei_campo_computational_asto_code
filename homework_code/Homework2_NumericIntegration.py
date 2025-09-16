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

# Importing necessary packages:

import numpy as np

# Beginning of function that will calculate the integral and plot it in the 
# xy plane:

def numerical_integration(start, end, steps, method):
    '''
    This function will calculate the integral of e^(t^2) over a user-inputted
    interval and steps. The function will calculate the integral with a 
    trapezoidal rule, Simpson's rule, and a Gauss, while also calculating the 
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
            "Simpson",
            "Gauss"

    
    Outputs:

    
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
    
    elif isinstance(method, str) == False:
        raise ValueError("The 'method' argument requires you to input a string." \
        "You inputted: " + str(method))
    
    # The code will now do more error checking for the 'method' argument. It 
    # will first capitalize the entire string to ensure that there are no 
    # string mismatches. For example, the code will not be case sensitive if 
    # a user inputs method = "trapezoidal" or method = "Trapezoidal". Once the
    # string is capitalized, it will then check to see if the inputted string is
    # and acceptable value by looking through a dictionary.


    # Creating dictionary of acceptable values for the 'method' arguement:

    method_dictionary = {'Integration Method': ['TRAPEZOIDAL', 'SIMPSON', 'GAUSS']}

    for vals in zip(method_dictionary):
        if method != method_dictionary.values[vals]:



    


    # Creating the exponential function that will be numerically integrated over
    # later:

    def function(x):
        return(np.exp(x**-2))
