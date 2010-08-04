"""
Evolutionary optimizations
"""


import numpy

from random import flip


def mutate(value, lower, upper, size):
    """
    Mutate value in the domain [lower,upper] with mutation 'size' as a decimal 
    percentage of the domain.
    WARNING:
        Doesn't work if the mutation size is too big and breaches both ends
        of the domain! 
    """
    
    # Define a random size for the mutation in the domain
    deltav = size*(upper - lower)
    
    vmax = value + deltav
    
    vmin = value - deltav

    if vmax > upper:
        
        # Percentage of the the total mutation size that fell in the domain
        bias = (upper - value)/deltav
        
        mutated = numpy.random.uniform(vmin, upper)
        
        # This compensated for the smaller region after 'value'
        while not (mutated >= value or flip(bias)):
        
            mutated = numpy.random.uniform(vmin, upper)
            
    elif vmin < lower:
        
        # Do the same as above but in the case where the mutation size breaches
        # the lower bound of the domain
        bias = (value - lower)/deltav
                        
        mutated = numpy.random.uniform(lower, vmax)
        
        # This compensated for the smaller region after 'value'
        while not (mutated <= value or flip(bias)):
        
            mutated = numpy.random.uniform(lower, vmax)
            
    else:
        
        # If the mutation size doesn't breach the domain bounds, just make a 
        # random mutation        
        mutated = numpy.random.uniform(vmin, vmax)
        
    return mutated


def mean_recomb(x, y):
    """
    Do a recombination of x and y using their mean
    
    Parameters:
    
        x: array like agent (estimate)
        
        y: array like agent (estimate)
    """
    
    recomb = 0.5*(x + y)
    
    return recomb


def wmean_recomb(x, x_fit, y, y_fit):
    """
    Do a recombination of x and y using a weighted mean.
    Weights are the respective fitness functions.
    
    Parameters:
        
        x: array like agent (estimate)
        
        x_fit: fitness of estimate x
        
        y: array like agent (estimate)
    
        y_fit: fitness of estimate y
    """
    
    recomb = (x_fit*x + y_fit*y)/float(x_fit + y_fit)
    
    return recomb



def recomb_pop(population, fitness, recomb, lower, upper):
    """
    Recombines the whole population according to a fitness function and a 
    recombination strategy.
    
    Parameters:
    
        population: list of elements in the population. Each element should be
            an Nx1 dimensional array like
            
        fitness: function object that can evaluate the fitness function given an
            element of the population
            
        recomb: function object of the recombination strategy
        
        lower: list with the lower bounds of the parameter space
        
        upper: list with the upper bounds of the parameter space
    """
    
    assert len(lower) == len(upper) == len(population[0]), \
        "upper and lower must have the same number of elements as there are" + \
        " dimensions in the problem"

    new_pop = []
    
    for i, element in zip(xrange(len(population)), population):
                
        if flip(fitness(element)):
            
            left_over = population[0:i] + population[i+1:]
            
            for mate in left_over:
                
                if flip(fitness(mate)):
                    
                    child = recomb(element, mate)

                    new_pop.append(child)
                    
                    if len(new_pop) == len(population):
                        
                        break
        
        if len(new_pop) == len(population):
            
            break
    
    while len(new_pop) != len(population):
        
        estimate = []
        
        for i in xrange(len(lower)):
            
            estimate.append(numpy.random.uniform(lower[i], upper[i]))
            
        new_pop.append(numpy.array(estimate))
    
    return new_pop
            
            
        