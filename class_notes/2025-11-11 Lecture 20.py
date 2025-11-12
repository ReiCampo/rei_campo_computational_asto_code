# Rei Campo
# Computational Methods


############################################################################
############################################################################
###                                                                      ###
###                           LECTURE 20 NOTES                           ###
###                                                                      ###
############################################################################
############################################################################

# As we evolve our equations, it becomes numerically unstable. This is not good!
# We need a method that can handle long term analysis. We can use the Von
# Neumann technique.

# We take Fourier transforms to express the Neumann technique. This stabalizes
# our equation, however we risk having growing modes! We need to make sure that
# our h values are <= (a^2) / (2D)

# Our h value should always be in between -1 and 1. As long as we have that,
# our modes won't grow and grow and grow.

# But sometimes, even if you don't get growing or decaying modes, you may end up
# with a stable, but non-physical system.

# We can instead use the Crank-Nicolson method!
# This takes the eigenvalues of the system and averages them. It just works?
# This is weird numerical method stuff that just ends up working.

# However let's turn to a completely different approach:
# Sometimes we have to conserve energy or do some type of conservation in the 
# system that we are trying to solve. All the other methods will not work out
# when we look over long periods of time.

# This is where the Spectral Method comes into play:
# We can use Fourier modes and no longer use numerical approximations of a 
# derivative, you just take the derivative! This is becuase Fourier series
# are easy to take the derivatives of!
# You should always use this! This is your best option! However there are 
# serious limitations for this to work. Notice that if your initial shape is 
# well approximated by your Fourier transform, then you're golden! There's
# little to no error! But if your function is not well approximated by Fourier
# tranforms and you ahve to chop it up a lot in order to use transforms, 
# then it's almost like you need an infinite amount of terms in order for this
# to fit well. 