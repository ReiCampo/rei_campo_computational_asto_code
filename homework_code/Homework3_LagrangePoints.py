# 9/22/2025
# Rei Campo
# Computational Methods in Astrophysics Homework 3


### ----------------------------------------------------------------------- ###
###                          Homework Exercise 3                            ###
###  Assuming circular orbits and that the Earth is much more massive than  ###
###   either the Moon or the satellite, show that the distance r from the   ###
###       from the center of the Earth to Lagrange Point 1 satisfies:       ###
###                     GM/r^2) - (Gm/(R-r)^2) = w^2 r                      ###
### ----------------------------------------------------------------------- ###

# Starting by importing necessary packages:

import numpy as np
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt

# Creating the function that will solve the equation numerically:

def lagrange_points(start_value, numerical_evaluation, units):
    '''
    This function will solve for the distance the Earth-Moon Lagrange Point 1 is
    from the center of the Earth. 

    Inputs:

        start_value (float):
            The starting value the user would like the numerical solver to start
            at.

        numerical_evaluation (str):
            A user-inputted value that defines which numerical evaluation to
            use. Acceptable inputs are:
            "Newtonian"
            "Secant"

        units (str):
            The user can choose what units they want their output to be in. The
            list of acceptable inputs are:
            "SI"
            "AU"
    

    Outputs:
        distance_to_L1 (float):
            This is the distance from the center of the Earth to the Lagrange
            Point 1 in the units specified by the user.

    '''

    ### Starting off with some error checking:

    if isinstance(numerical_evaluation, str) == True:

        # Capitalizing the string the user inputted:

        numerical_evaluation = numerical_evaluation.upper()

        if numerical_evaluation not in ("NEWTONIAN", "SECANT"):
            raise ValueError("Your 'numerical_evaluation' input is not a value" \
            "that is recognized by the function. " \
            "You inputted: " + str(numerical_evaluation) + ".  Please use one"
            "of the following allowable inputs:\n"
            "'Newtonian'\n"
            "'Secant'") 
        
    elif isinstance(numerical_evaluation, str) == False:
        raise ValueError("The 'numerical_evaluation' input requires a string " \
        "input. You have entered: " + str(numerical_evaluation) + ".  Please "
        "enter one of the acceptable values:\n"
        "'Newtonian'\n"
        "'Secant'")
    
    if isinstance(units, str) == True:

        # Capitalizing the string the user inputted:

        units = units.upper()

        if units not in ("SI", "ERG"):
            raise ValueError("Your 'units' input is not a value that is"
            "recognized by the function. You inputted: " + str(units) + ".  "
            "Please enter one of the acceptable values:\n"
            "'SI'\n"
            "'erg'")

    elif isinstance(units, str) == False:
        raise ValueError("The 'units' input requires a string input. You have " \
        "entered: " + str(units) + ".  Please enter one of the acceptable "
        "values:\n"
        "'SI'"
        "'erg'")


    ### First defining some local variables to the function:

    # Univeral Gravity in m^3 / (kg * s^2)
    G_univ = 6.6743e-11

    # Mass of Earth in kg
    M_Earth = 5.972167867791379e+24

    # Mass of the Moon in kg
    m_moon = 7.384e22

    # Angular velocity of Moon and satellite in 1/s
    w = 2.662e-6

    # Distance from center of Earth to center of Moon in m
    R_EM = 3.844e8


    # Creating a fucntion that defines the equation:

    def lagrange_equation(r_lagrange):

        # I am rearranging the function so that it is set to 0:
        func = ((G_univ * M_Earth) / r_lagrange**2) - ((G_univ * m_moon) / (R_EM - r_lagrange)**2) - (w**2 * r_lagrange)

        return func
    
    distance_to_L1 = 0
    
    ### Starting with Newtonian method first: 

    if numerical_evaluation == "NEWTONIAN":

        # I have decided to let the user input the h value later instead of 
        # making it an argument to the function. Below, the user is prompted
        # to enter in an h value which will be used later when calculating the
        # derivative:
        
        h_value = float(input("You have selected the Newtonian method to numerically " \
            "solve for the Lagrangian Point. Please type in an (h) value to " \
            "calculate the derivative using the central difference method. A " \
            "recommended (h) value is 1e-5."))

        # Using the central difference method to find the derivative for the
        # function:

        def newtonian_derivative(h): ### Need to fix this later

            df = (lagrange_equation(start_value + (h/2)) - lagrange_equation(start_value - (h/2))) / h

            return df
        
        # Creating an initializer to be used in the while loop:

        newtonian_initializer = 0

        # Setting up the while loop:

        while newtonian_initializer < 1000:

            newton_guess = start_value

            # Calculating the guess:

            newton_calculation = newton_guess - (lagrange_equation(newton_guess) / newtonian_derivative(h_value))

            # I chose 1e-10 arbitrarily since that the error of the root being
            # 1e-10 off the true value is allowable:
             
            if abs(newton_calculation) < 1e-10:
                # Update the distance_to_L1 variable to the calculated value and
                # break from the loop: 
                distance_to_L1 = newton_calculation
                break

            else:
                # Update the newton_guess variable with the new calculation and
                # update the initializer: 
                newton_guess = newton_calculation
                newtonian_initializer = newtonian_initializer + 1

    # If not using Newtonian method, move on to secant:

    elif numerical_evaluation == "SECANT":

        second_value = float(input("You have selected the Secant method to" \
            "numerically solve for the Lagrangian Point. To use this method," \
            "the calculation needs two starting values. Your input for" \
            "start_value will be used as the first point, however you must " \
            "choose another point. It is recommended to select a second value" \
            "that is relativley close to your start_value."))

        def secant_point_value(second_point):

            x3 = second_point - lagrange_equation(second_point) * ((second_point - start_value) / (lagrange_equation(second_point) - lagrange_equation(start_value)))

            return x3
        
        secant_initializer = 0

        while secant_initializer < 1000:

            secant_guess1 = start_value 

            secant_guess2 = second_value

            secant_calculation = secant_point_value(secant_guess1, secant_guess2)

            if abs(secant_calculation) < 1e-10:
                distance_to_L1 = secant_calculation
                break
            else:
                secant_guess1 = secant_guess2
                secant_guess2 = secant_calculation
                secant_initializer = secant_initializer + 1

    
    return distance_to_L1