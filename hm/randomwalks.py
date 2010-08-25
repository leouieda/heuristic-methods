"""
Set of random walks for different kinds of problems (combinatorial, genetic
algorithm, continuous, etc.)
"""

import numpy

import hm.utils


def continuous(how_many, ndims, lower, upper):
    """
    Generate random points in the parameter space.
    
    Parameters:
    
        how_many: how many points to generate
        
        ndims: number of dimensions in the problem 
        
        lower: list with lower bounds for the parameter space
        
        upper: list with upper bounds for the parameter space
        
    Returns:
    
        list with generated points
    """
    
    assert len(lower) == ndims and len(upper) == ndims, \
        "Must have 'ndims' values in both 'lower' and 'upper'"
    
    population = []
    
    for i in xrange(how_many):
        
        individual = []
        
        for j in xrange(ndims):
            
            param = numpy.random.uniform(lower[j], upper[j])
            
            individual.append(param)
    
        population.append(individual)
        
    return population

    
def genetic(pop_size, nbits):
    """
    Generate a random population for use in Genetic Algorithms.
    Each individual is a list of ones and zeros that describes its chromosome.
    
    Parameters:
        
        pop_size: size of the population
        
        nbits: number of bits in the chromosomes
        
    Returns:
    
        list of individuals
    """
    
    population = []
    
    for i in xrange(pop_size):
        
        individual = []
        
        for bit in xrange(nbits):
            
            # Flip returns 1 or 0 according to a probabilty. So sending 0.5
            # returns 1 or 0 with 50-50 chance
            individual.append(hm.utils.flip(0.5))
            
        population.append(individual)
        
    return population


def salesman(pop_size, ncities):
    """
    Generate random routes for the traveling salesman problem.
    Always starts and ends in city 0
    
    Parameters:
    
        pop_size: how many routes to generate
        
        ncities: number of cities to visit
        
    Returns:
    
        list of routes. Each route is a list with the city indexes in the order
        visited
    """
    
    population = []
    
    for i in xrange(pop_size):        
        
        buffer = range(1, ncities)
        
        route = [buffer.pop(numpy.random.randint(N)) \
                 for N in xrange(ncities - 1, 0, -1)]
            
        population.append(route)
        
    return population
            
            
            