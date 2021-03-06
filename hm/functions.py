"""
Some functions to try out the optimizers with. 
"""

import numpy


def twomin(x):
    
    return (x + 0.5)**4 - 50*(x**2) - 20*x


def eggbox(x, y):
    
    return 200*numpy.sin(x + 1.5*numpy.pi)*numpy.cos(y) + x**2 + y**2