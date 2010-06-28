"""
Factorial calculators: recursive, iterative, Stirling's approximation and
prime number factorization.
"""

import math

import primate


def rfactorial(x):
    """
    Recursive factorial function
    """
    
    if x == 1:
        
        return 1

    else:
        
        return x*rfactorial(x - 1)


def ifactorial(x):
    """
    Iterative factorial function
    """

    res = 1

    for i in range(2, x + 1):
        
        res *= i

    return res



def afactorial(x):
    """
    Approximate the factorial using Stirling's formula
    """

    return math.e**(x*math.log(x) - x)


def pfactorial(x):
    """
    Calculate the factorial using prime number decomposition
    """

    primes = primate.primes(x)

    pfactors = [0 for i in xrange(len(primes))]

    # Calculate the prime factors of all the numbers from 2 to x
    for i in xrange(2, x + 1):

        tmp = primate.factor(i, primes)
        
        # Sum the factors to the ones of the previous numbers
        for j in xrange(len(pfactors)):
            
            pfactors[j] += tmp[j]

    # Evaluate the result from the factors
    res = 1

    for i in xrange(len(primes)):
        
        res *= primes[i]**pfactors[i]

    return [res, pfactors]

