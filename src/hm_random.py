"""
Random operations
"""

import numpy
import math


def normal_rand(mean, stddev):
    """
    Return sample from a Normal distribution.
    """
    
    u1 = numpy.random.uniform(-1, 1)
    
    u2 = numpy.random.uniform(-1, 1)
    
    r = u1**2 + u2**2
    
    while r >= 1.0:

        u1 = numpy.random.uniform(-1, 1)
    
        u2 = numpy.random.uniform(-1, 1)
    
        r = u1**2 + u2**2    

    return mean + stddev*u2*math.sqrt(-2*math.log(r)/r)
    

def flip(bias):
    """
    Flips a biased coin.
    Returns 1 for heads, 0 for tails 
    
    Parameters:
    
        bias: probability of returning heads
    """
    
    res = numpy.random.random()
        
    if res > bias:
        
        return 0
    
    else:
        
        return 1

