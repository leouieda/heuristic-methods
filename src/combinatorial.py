"""
Solvers for the shortest path in a graph.
Depends on graph.Graph class.
"""

import numpy

def randint_individual(size):
    """
    Make a random individual with integer values from 0 to size
    """
    
    index_buffer = range(size)
    
    individual = []
    
    while len(index_buffer) > 0:
        
        index = numpy.random.randint(len(index_buffer))
        
        individual.append(index_buffer.pop(index))
        
    return individual
    