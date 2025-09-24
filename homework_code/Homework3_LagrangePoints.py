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
from astropy import units as u
import argparse

# Creating the function that will solve the equation numerically:

def lagrange_points(start_value, numerical_evaluation, units = "SI", error_tolerance = 1e-5):
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
            "Kilometers"
            "Centimeters"
            "Milimeters"

        error_tolerance (float):
            The allowable error tolerance the numerical calculation of the non-
            linear equation. It is automatically set to 1e-10
    

    Outputs:
        lagrange_point_distance (str):
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

        if units not in ("SI", "AU", "KILOMETERS", "CENTIMETERS", "MILIMETERS"):
            raise ValueError("Your 'units' input is not a value that is "
            "recognized by the function. You inputted: " + str(units) + ".  "
            "Please enter one of the acceptable values: \n"
            "'Meters' \n"
            "'AU' \n"
            "'Centimeters' \n"
            "")

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
            "recommended (h) value is 1e-5. "))

        # Using the central difference method to find the derivative for the
        # function:

        def newtonian_derivative(h): 

            df = (lagrange_equation(start_value + (h/2)) - lagrange_equation(start_value - (h/2))) / h

            return df
        
        # Creating an initializer to be used in the while loop:

        newtonian_initializer = 0
        
        newton_guess = start_value

        # Setting up the while loop:

        while newtonian_initializer < 1000:

            # Calculating the guess:

            newton_calculation = newton_guess - (lagrange_equation(newton_guess) / newtonian_derivative(h_value))
             
            if abs( (newton_calculation - newton_guess) / newton_calculation) < error_tolerance:
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

        # User is prompted to input a second value so the secant method can be
        # calculated: 

        second_value = float(input("You have selected the Secant method to " \
            "numerically solve for the Lagrangian Point. To use this method, " \
            "the calculation needs two starting values. Your input for " \
            "start_value will be used as the first point, however you must " \
            "choose another point. It is recommended to select a second value " \
            "that is relativley close to your start_value. Your start_value " \
            "input was: " + str(start_value) + ". "))
        
        # Creating the function that will be used to calculate the third point:

        def secant_point_value(first_point, second_point):

            x3 = second_point - lagrange_equation(second_point) * ((second_point - first_point) / (lagrange_equation(second_point) - lagrange_equation(first_point)))

            return x3
        
        # Creating an initializer to be used later in the while loop:
        
        secant_initializer = 0

        # Creating starting values that will be updated later by the while
        # loop: 

        secant_guess1 = start_value 

        secant_guess2 = second_value

        # Setting up the while loop:

        while secant_initializer < 1000:

            # Calculating the third point:

            secant_calculation = secant_point_value(secant_guess1, secant_guess2)

            # I chose 1e-10 arbitrarily since that the error of the root being
            # 1e-10 off the true value is allowable:

            if abs( (secant_calculation - secant_guess2) / secant_calculation) < error_tolerance:
                # If the value is small enough, distance_to_L1 will be updated
                # to the secant_calculation and the while loop will be broken 
                # out of: 
                distance_to_L1 = secant_calculation
                break

            else:
                # Loop will continue the calculation and update the guesses and
                # the initializer:
                secant_guess1 = secant_guess2
                secant_guess2 = secant_calculation
                secant_initializer = secant_initializer + 1

    # Converting the distances to what the user inputted:

    distance_to_L1 = distance_to_L1 * u.meter

    if units == "AU":
        distance_to_L1 = format(distance_to_L1.to(u.au), ".6")
    elif units == "KILOMETERS":
        distance_to_L1 = format(distance_to_L1.to(u.km), ".6e")
    elif units == "CENTIMETERS":
        distance_to_L1 = format(distance_to_L1.to(u.cm), ".6e")
    elif units == "MILIMETERS":
        distance_to_L1 = format(distance_to_L1.to(u.mm), ".6e")
    else:
        distance_to_L1 = format(distance_to_L1, ".6e")

    
    
    lagrange_point_distance = "The approximate distance of the Earth-Moon " \
    "Lagrange point 1 using the " + str(numerical_evaluation) + " method is: " \
    + str(distance_to_L1)

    return lagrange_point_distance



# Creating parser so the code can be called on the command line:

parser = argparse.ArgumentParser(description = "Calculate the Earth-Moon " \
"Lagrangian Point 1 using either the Newtonian or Secant numerical evaluation " \
"method for non-linear equations.")

# Creating the start parser
parser.add_argument("start_value", 
                    type = float, 
                    help = "This is the initial guess value that will be used " \
                    "in both the Newtonian and Secant methods.")

# Creating the end parser
parser.add_argument("numerical_evaluation", 
                    type = str,
                    help = "This is the numerical calculation method that will " \
                    "be used to solve the non-linear equation that determines " \
                    "the distance of the Earth-Moon Lagrange Point 1. The " \
                    "acceptable values are: \n" \
                    "'Newtonian' \n" \
                    "'Secant'")

# Creating the steps parser
parser.add_argument("--units",
                    default = "SI",
                    type = str,
                    help = "The units of the distance to from the center of the " \
                    "Earth to the L1 point. If no units are specifie, SI are " \
                    "automatically used, however acceptable inputs are: \n" \
                    "'AU' \n" \
                    "'Kilometers' \n" \
                    "'Centimeters' \n" \
                    "'Milimeters'")

# Creating the method parser
parser.add_argument("--error_tolerance",
                    default = 1e-10,
                    type = float,
                    help = "The amount of tolerable error the numerical " \
                    "calculation can be off when finding the root of the " \
                    "non-linear equation. It is automatically set to 1e-10.")

args = parser.parse_args()

L1_distance = lagrange_points(args.start_value,
                              args.numerical_evaluation,
                              args.units,
                              args.error_tolerance)

print(L1_distance)
