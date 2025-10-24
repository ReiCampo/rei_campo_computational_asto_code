# 10/21/2025
# Rei Campo


############################################################################
############################################################################
###                                                                      ###
###                           LECTURE 14 NOTES                           ###
###                                                                      ###
############################################################################
############################################################################


##---------------------------------------------------------------
##              Continuation of Monte Carlo lecture             -
##---------------------------------------------------------------

# Statistical mechanics: Importance sampling isn't a fancy trick just for Monte
# Carlo integration. We can use it in Stat Mech! Think about the problem of 
# calculating the average value of a quantity in a physical system in thermal
# equilibrium. We can use Monte Carlo to sample random N states as N gets larger
# we will get closer to the correct value.

# However, in the case of the Boltzmann probability, it's exponential, so that 
# means that the vast majority of the states of exponentially small 
# probabilities. So most of our randomly chosen energies contribute almost 
# nothing to this sum.

# But we can use importance sampling in order to get the important ones in our 
# sum. 

# In the equation to calculate the average of X, the w is the weight of the
# number.

# Importance sampling is just physically modeling what the system might do in 
# and actual physics experiment. We are trying to model the random changes of
# the system, just like any physics experiment!

# However, finding the normalization for the system is hard and you can't always
# find the Z value! You can use Markov Chains in order to figure out the states
# and then sum to get an average. If you choose yoru transition probabilities
# the right way, we can insure that the chance of ending up at any one state on
# the chain is the Boltzmann probability

# So, if you start in the system in any random state and let the chain run long
# enough, then the distribution over the states will converge to the Boltzmann
# distribution

# The distribution of the entire chain is the distribution of the system!
# However, even though the markov chain always converges to the correct Boltzman
# distribution, we don't know how many steps this will actually take!! There is 
# no general way to know how many steps you will need to take to get to the
# converged answer.

# However, there are some problems where you want to find the global minimum,
# but the previous methods we used cannot find a global minimum! (Bisection, 
# Newton's method, etc.) 


##----------------------------------------------------------------
##          Moving on to Ordinary Differential Equations         -
##----------------------------------------------------------------

# Remember, to solve ODE's is basically "guess, and maybe the solution will 
# work out!" Boundry conditions become very different in numerical techniques
# than in other situations. 