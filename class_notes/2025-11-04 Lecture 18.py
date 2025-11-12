
############################################################################
############################################################################
###                                                                      ###
###                           LECTURE 18 NOTES                           ###
###                                                                      ###
############################################################################
############################################################################


##----------------------------------------------------------------
##                Partial Differential Equations                 -
##----------------------------------------------------------------

# These are the main equatiosn that go into solving physical systems. Nature
# prefers to use second order PDEs. Just like ODE's, PDEs have issues with 
# boundary conditions. However boundary value PDEs are easier to solve in PDEs
# than in ODEs

# Boundary Value PDEs:
# Let's take an electric circuit in 2D. We can divide the space of our circuit
# into a grid. However, we have to mindful about how many points we want to put 
# in this grid! It can slow down run time significantly! 

# We can use the central difference theorem in order to find the 

# Jacobi Method:
# We can use the Jacobi method in order to use the relaxation method. We just
# keep chosing values over and over and over until the fixed values begin to
# settle down and relax.
# Remember, we have to worry if it will converge! 
# The shortcomings of this method is the actual geometry of the problem you are 
# trying to solve. Rectangles and squares work great, even transforming to polar
# coordinates to solve the circle can be good too. But you are limited to the 
# actual geometry of the problem.


##---------------------------------------------------------------
##                          Excercise 1                         -
##---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

M = 100
V = 1.0
target = 1e-6
a = 1
epsilon = 1

phi = np.zeros([M + 1, M + 1], float)
phiprime = np.empty([M + 1, M + 1], float)
rho = np.zeros([M + 1, M + 1], float)
rho[20:40, 20:40] = -V
rho[60:80, 60:80] = V


delta = 1.0
while_counter = 0

while delta > target:
    for i in range(M + 1):
        for j in range(M + 1):
            if i == 0 or i == M or j == 0 or j == M:
                phiprime[i, j] = phi[i, j]
            else:
                phiprime[i, j] = ((phi[i + 1, j] + phi[i - 1, j] + phi[i, j+1] + phi[i, j - 1]) / 4) + ((a**2 / (4 * epsilon)) * (rho[i, j]))
                
    delta = max(abs(phi - phiprime))
    
    phi, phiprime = phiprime, phi
    
    if while_counter > 1e5:
        break
    else:
        print(while_counter)
        while_counter += 1
        
fig, ax = plt.subplots()
plt.imshow(phi, origin = 'lower')
