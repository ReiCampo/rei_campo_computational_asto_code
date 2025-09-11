### Lecture 6 Notes
### 9/11/2025

# Adaptive Integration:
# Nesting integration is powerful because you can use the previous calculations
# you made and double the amount (so you already know the end points of the 
# previous calculation, now you are getting the midpoints if you, for example
# double the amount of points)

# There are some basic numerical integrations tht you can use:
# Romberg integration is a step up above adaptive integration. This s because
# you can use your error that you get from calculating the numerical integral
# and adust your calculation over the integral.
# Using this method improves the accuracy of your calculation with little to
# no computational cost as if you were just calculating Simpson's rule, for 
# example.

# Higher order methods:
# We can use a higher order polynomial (above 2) by using Newton-Cotes method

# Gaussian Quadrature:
# You can use methods where you can have a nonuniform set of N points. (Aka: 
# not evenly spaced). When you calculate the "weight" of all the points, you 
# only have to do this once unlike with Simpson's and Trapezoidal rule.


#########################################
###             Excersie 1            ###
#########################################

# Downloading these functions from:
# https://public.websites.umich.edu/~mejn/computational-physics/gaussxw.py


from numpy import ones,copy,cos,tan,pi,linspace

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

#########################################################################



def gaussian_quadriture(N, a, b):

    def function(x):
        return(x**4 - 2*x + 1)
    
    x_val, weight = gaussxwab(N, a, b)
    fun = 0
    for i in range(0, N):
        fun += weight[i] * function(x_val[i]) 
    
    return fun

test = gaussian_quadriture(3, 0, 2)
print(test)


# Choosing an integration method:
# Generally, higher order integration methods are going to be better IF the 
# function is well behaved!!! Basically, if it looks like a recognizable 
# function, it's going to be well behaved.
# If this doesn't work, go lower order (like Simpsons for example)

# How to deal with infinite range intervals:
# You can subsitute a function that can go to infinity when you use certain
# numbers.



