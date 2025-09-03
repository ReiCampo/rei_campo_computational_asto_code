# 9/2/20205
# Rei Campo
# Computational Methods in Astrophysics

###-------------------------------------------------------------------------###
### Homework Excersise 1: Take the in-class excercise of creating a fucntion
###                       that calculates the time it takes a ball to fall down
###                       to the ground and use argparse instead of argv.
###-------------------------------------------------------------------------###

# Importing necessary packages
import argparse

# Starting by creating the function

def drop_time(height, gravity = 'Earth', v_0 = 0.0):
    """
    This function calculates the time it takes for an object to touch the ground
    at a user-inputted height. The function inherently assumes the inputs are in
    meters and seconds, there is no initial velocity, and that we're on Earth. 
    However, you can adjust the acceleration due to gravity by simply entering 
    in a number or you can even use common strings like 'Venus,' 'Mars,' or 
    'Moon.'

    Inputs:
        height (float):  
            This takes in a non-negative number. The units are in meters.

        gravity (float, string):  
            This can either take a float or string. The units are in 
            meters/second^2 and gravity must be a negative number. The string 
            input has to match one of the known keywords. If the user inputs an
            unrecognized string, they will be prompted to enter in an acceptable
            keyword. Keywords that can be used are:
            "Sun"
            "Mercury"
            "Venus"
            "Earth"
            "Moon"
            "Mars"
            "Jupiter"
            "Saturn"
            "Uranus"
            "Neptune"
        
        v_0 (float):
            The initial velocity of the object. The function assumes there is 0
            initial velocity, and the units are in meters/second.
        
    Outputs:
        final_time (str):
            Outputs the time it took the object to reach the ground while also
            printing out the inputs you used. The units for time are seconds.

    """

    # Setting up some error conditions and string checking:

    if height < 0:
        raise ValueError("The 'height' value should be positive. You entered: " +  str(height))
    if isinstance(gravity, str):
        gravity = gravity.upper()
    elif gravity >= 0:
        raise ValueError("The 'gravity' arguement should be negative. You entered:" + str(gravity))
    
    # Setting up key words for the gravity argument:

    if gravity == "EARTH":
        gravity = -9.8
    elif gravity == "SUN":
        gravity = -274
    elif gravity == "MERCURY":
        gravity = -3.7
    elif gravity == "VENUS":
        gravity = -8.87
    elif gravity == "MOON":
        gravity = -1.625
    elif gravity == "MARS":
        gravity = -3.71
    elif gravity == "JUPITER":
        gravity = -24.79
    elif gravity == "SATURN":
        gravity = -10.44
    elif gravity == "URANUS":
        gravity = -8.69
    elif gravity == "NEPTUNE":
        gravity = -11.15
    else:
        raise ValueError("You inputted: " + str(gravity) + " as the argument \n"
        "for 'gravity'. This input is not a key word that can be used in the \n"
        "function.\n"
        "------------------------------\n"
        "The acceptable values are: \n"
        "'Earth',\n"
        "'Sun', \n"
        "'Mercury',\n"
        "'Venus',\n"
        "'Moon',\n"
        "'Mars',\n"
        "'Jupiter',\n"
        "'Saturn',\n"
        "'Uranus', and \n"
        "'Neptune'\n"
        "Check your spelling or use a value listed above.")



    # Since I am including initial velocities in the code, I will be using the
    # quadratic formula in order to solve the distance equation. I will always
    # be taking the negative of the quadratic formula to ensure that the time
    # will always be positive

    time = (v_0 - (v_0**2 - 2* gravity *height)**(1/2)) / gravity

    final_time = "The object took approximately " + str(round(time, 2)) + " seconds to reach the ground." + "\n" + "The inputs you used were:\n" + "Height: " + str(height) + " m" + "\n" + "Gravity: " + str(gravity) + " m/s^2" + "\n" + "Initial velocity: " + str(v_0) + " m/s"
    return final_time

# Writing in a main statement so the code can be ran from the terminal

# Setting up the parsing to be used later in the terminal
parser = argparse.ArgumentParser(description = "Calculate the time it takes an object to fall")

# Creating the height parser
parser.add_argument("--height", type = float, help = "This is the initial height from the ground that you are using the calculate the time it takes to fall. The units must be in meters.")

# Creating the gravity parser
parser.add_argument("--gravity", type = str, help = "This is the acceleration due to gravity in the units of meters/second^2. It automatically assumes Earth's gravity, however if you use your own input, it must be negative. There are some keywords that you can use like 'Earth' or 'Jupiter' that will automatically fill in the acceleration due to gravity for you. Check out the doc strings to see what key words you can input here.")

# Creating the initial velocity parser
parser.add_argument("--v_0", type = float, help = "This is the initial velocity in the units meters/second. It automatically assumes 0 initial velocity, however you can input positive or negative velocities. Reminder: a negative velocity is going in the direction of gravity (downward) while positive velocity is going opposite of gravity (upward).")

args = parser.parse_args()
print(args.height, args.gravity, args.v_0)

