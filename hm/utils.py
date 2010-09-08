"""
Misc utils
"""

import numpy
import math


def sort_by_fitness(population, fitness):
    """
    Sort the population in-place by each individual's fitness value.
    
    Parameters:
    
        population: list of individuals (usually also lists themselves)
        
        fitness: list with the fitness value of each individual in population
    """
    
    assert len(population) == len(fitness), \
        "Must have a fitness value for each individual."
    
    # [::-1] is to revert the array in order to sort by decreasing fitness
    sort_index = numpy.argsort(fitness)[::-1]    
    
    oldpop = numpy.copy(population)
    
    oldfit = numpy.copy(fitness)
    
    for i in xrange(len(population)):
        
        population[i] = oldpop[sort_index[i]].tolist()  
        
        fitness[i] = oldfit[sort_index[i]]   
        
        

def best_by_fitness(population, fitness):
    """
    Get the best individual in the population based on its fitness value.
    
    Parameters:
    
        population: list of individuals (usually also lists themselves)
        
        fitness: list with the fitness value of each individual in population
    """
    
    assert len(population) == len(fitness), \
        "Must have a fitness value for each individual."
        
    best = fitness.index(max(fitness))
    
    return population[best]        
        
    
def flip(bias):
    """
    Flips a biased coin.
    Returns 1 for heads, 0 for tails 
    
    Parameters:
    
        bias: probability of returning heads
    """
    
    assert 1 >= bias >= 0, \
        "Bias must be with range [0,1]"
    
    res = numpy.random.random()
        
    if res > bias:
        
        return 0
    
    else:
        
        return 1


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