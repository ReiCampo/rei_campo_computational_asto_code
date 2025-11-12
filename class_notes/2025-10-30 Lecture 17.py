# 10/30/2025
# Rei Campo

# HAPPY ALMOST HALLOWEEN!!!!!!!!!!!!!!!


############################################################################
############################################################################
###                                                                      ###
###                           LECTURE 17 NOTES                           ###
###                                                                      ###
############################################################################
############################################################################

##---------------------------------------------------------------
##                Things to Know for the Homework               -
##---------------------------------------------------------------

# You will need to use plotly for this homework. Play around with it, have fun!

# plotly.express does a quick plot in a simple command, however if you want to
# be more detailed, you'll have to use regular plotly.

# Everything in plotly uses dictionaries, so keep that in mind when working with
# it!

# Plotly has really great data manipulation just like R!


##---------------------------------------------------------------
##                        Continuing ODEs                       -
##---------------------------------------------------------------

# The verlet method is a modification of the leapfrog method.


# The modified midpoint method: 
# The other advantage of the leapfrog method is that the total error after 
# summing all steps is an even function of the step size h

# This is because the leapfrog method is time symmetric! So as we go to each
# step, we should get the same error for each step, just opposite signs.

# We do all of this to do the Burlisch-Stoer Method. The modified midpoint
# method is rarely used alone since it doesn't do anything better than the 
# leapfrog method. You can combine it with the Richardson extrapolation and it
# becomes the powerful Burlisch-Stoer method.

# This becomes a free higher order calculation. Better accuracy than previous
# methods. You get a better approximation for your ODE.

# This solver is adaptive in its nature too. It is efficient in selecting the 
# step sizes you need to get a good approximation for the solutions of the ODE.

# This is even better than 4th order Runge-Kutta, but it is more difficult to
# code. So it depends what you are looking for!

# Note: This method only works if the error converges fairly quickly with each
# step.

# For this method, you really only want to go to 8 - 10 steps at each interval
# H. So choosing the size of H is going to be important here.

# You can always test out an H value then you can adjust the value so you can
# get a better interval.


# Stiff ODEs
# There are ODEs that still have problems when trying to solve it. This means 
# that they do not converge well (aka: stiff ODEs)

# Sometimes you can solve these equations on a small enough time scale. But you
# have to know this ahead of time!

# So what can we do with these problems? Well, we can use implicit solvers. We
# can use the backwards Euler equations.

# Everything we've been talking about, we knew all the initial values, so we 
# were able to propogate our solution forward. None of these techniques would 
# work if we didn't have the starting conditions.

# Boundary Value Problems:

# One technique: The Shooting Method
# Basically, you start with random initial conditions and figure it out from
# there. You can keep honing in on the initial conditions until you get the
# target that you want.

# Another method: Relatxation Method
# You start with one guess and let the solution relax until we get to the answer
# we are looking for.

# Eigenvalue Problems: