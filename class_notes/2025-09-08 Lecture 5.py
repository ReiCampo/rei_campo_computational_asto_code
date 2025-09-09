### Lecture 5 Notes
### 9/8/2025

# Finishing up exercise 6 from last class:

#########################################
###             Excersie 1            ###
#########################################

def trap_rule(a, b, n):

    def function(x):
        return x**4 - 2*x + 1
    
    delta_x = (b - a) / n

    s = 0.5 * function(a) + 0.5 * function(b)

    for k in range(1, n):
        s += function(a + k*delta_x)
    
    return delta_x*s

test_1 = trap_rule(0, 2, 10)

print(test_1)
# answser: 4.50656

# simpson's rule is going to be more accurate than trapezoidal rule

#########################################
###             Excersie 2            ###
#########################################

def simpsons_rule(a, b, n):

    def function(x):
        return x**4 - 2*x + 1
    
    delta_x = (b - a) / n

    s = function(a) + function(b)

    for k in range(1, int(n / 2) + 1):
        s += 4 * function(a + (2*k - 1)*delta_x)

    
    for j in range(1, int(n/2)):
        s += 2 * function(a + 2 * j * delta_x)

    return (delta_x / 3) * s


ten_slices = simpsons_rule(0, 2, 10)
print(ten_slices)
#printed answer: 4.400426666666667

hundred_slices = simpsons_rule(0, 2, 100)
print(hundred_slices)
#printed answer: 4.400000042666668


# Remember, all these numerical techniques contain error! Approximation error is
# the dominant source of error. This is the error introduced because linear or
# quadratic fit only approximates the true function.

# Error of trapezoidal rule is e = (1/12)(h^2)|f'(a) - f'(b)|
# Error for Simpson's rule is e = (1/90)(h^4)|f'''(a) - f'''(b)|

# estimates of the error in our integration schemes depend on knowing the 
# derivatives at a and b. 

# we can estimate the error by evaluating the integral with a certain number of 
# steps N and step size h. then we can evaluate the integral again with let's 
# say double the number of steps N_2 = 2N and half the step size h_2 = (1/2)h



#########################################
###             Excersie 3            ###
#########################################

# I'm going to call my previous functions from the other exercises.

simp_10 = simpsons_rule(0, 2, 10)
print(simp_10)

simp_20 = simpsons_rule(0, 2, 20)
print(simp_20)

simpson_error = (1/15) * (simp_20 - simp_10)
print(simpson_error)
#printed answer: -2.6666666666604518e-05


# But what is the correct amount of steps we should use? More steps isn't always
# worth it b/c it takes computational time. Choosing too small of h causes the 
# round off error to grow. You can even choose an h so so so small that when you
# add it to the next step to our sum, it won't actually increase it b/c of
# machine precision!

# For Python, the "sweet spot" is 10,000. Going beyond this number will actually
# make our answser worse! This is great because running Simpson's rule for 10k
# steps is basically instantaneous.

############################
### Adaptive Integration ###
############################

# You start with some N (10), and then you calculate N = 20. You then check the
# error 

