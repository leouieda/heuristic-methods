"""
Solvers for the shortest path in a graph.
Depends on graph.Graph class.
"""

import math
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


def distance((lat1, lon1), (lat2, lon2)):
    """
    Compute the distance between the coordinates lat1,lon1 and lat2,lon2
    Give coordinates in radians
    """
    
    earth_radius = 6371.
    
    cos_arc = math.sin(lat1)*math.sin(lat2) + \
              math.cos(lat1)*math.cos(lat2)*math.cos(lon1 - lon2)
                    
    distance = math.sqrt(2*(earth_radius**2)*(1 - cos_arc))
    
    return distance


def total_distance(path, cities):
    """
    Calculate the total distance of the path given a list of cities with 
    (lat,lon) coordinate pairs (in radians)
    """
    
    assert len(path) == len(cities), "Path must pass through all cities"
    
    total = 0
    
    size = len(path)
    
    for i in xrange(size):

        total += distance(cities[path[i]], cities[path[(i + 1)%size]])
        
    return total

    
def salesman_random_walk(popsize, ncities):
    """
    Generates a population of 'popsize' solutions to the travelling salesman
    problem with ncities 
    """
    
    population = []
    
    for i in xrange(popsize):
        
        population.append(randint_individual(ncities))
        
    return population


def salesman_fitness(population, cities):
    """
    Fitness criterium for the salesman problem. Returns the fitness of each
    individual in a population given a list of cities with (lat,lon) coordinate 
    pairs (in radians)
    """
    
    distances = [total_distance(guy, cities) for guy in population]
    
    best = max(distances)
    
    worst = min(distances)
    
    fitness = [(dist - worst)/(best - worst) for dist in distances]
    
    return fitness


def salesman_evolutive_pressure(population, fitness, percent_survivors, \
                                ncities):
    """
    Kill some people in the population based on their fitness.
    """
    
    assert len(population) == len(fitness), \
        "Must have a fitness value for each individual in the population"

    assert percent_survivors >=0 and percent_survivors <= 1, 
        "percent_survivors must be a decimal percentage from [0,1]"

    popsize = len(population)
    
    nsurvivors = int(popsize*percent_survivors)
    
    # Sort the population by fitness
    sort_index = numpy.argsort(fitness)
        
    for i in xrange(popsize):
        
        tmp = population[i]
        population[i] = population[sort_index[i]]
        population[sort_index[i]] = tmp
        
    population[nsurvivors:] = salesman_random_walk(popsize - nsurvivors, \
                                                   ncities)
    
    return population
    