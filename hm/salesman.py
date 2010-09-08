"""
Solvers of the Traveling Salesman problem
"""

import math

import numpy

import hm.utils
import hm.genetic


earth_radius = 6371.


def build_distance_table(cities, type='cartesian'):
    """
    Build the distances table.
    
    Parameters:
    
        cities: list with the coordinates of each city (usually x,y tuples)
        
        type: kind of coordinates. Can be 
                * cartesian = (x,y)
                * geographic = (lon,lat) in decimal degrees
                
    Returns:
    
        the distances table
    """
    
    assert type in ['geographic', 'cartesian'], \
        "Invalid coordinate type '%s'" % (type)
    
    d2r = math.pi/180.
    
    ncities = len(cities)
    
    table = numpy.zeros((ncities, ncities))
    
    for i in xrange(ncities):
                
        for j in xrange(i + 1, ncities):

            xi, yi = cities[i]
            
            xj, yj = cities[j]
            
            if type == 'cartesian':

                distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                
            if type == 'geographic':
                                
                cos_arc = math.sin(d2r*yi)*math.sin(d2r*yj) + \
                          math.cos(d2r*yi)*math.cos(d2r*yj)* \
                          math.cos(d2r*(xi - xj))
                
                distance = math.sqrt(2*(earth_radius**2)*(1 - cos_arc))

            table[i][j] = distance
            
            table[j][i] = distance            
            
    return table


def total_distance(route, dist_table):
    """
    Calculate the distance of a route.
    
    Parameters:
    
        route: list with the city indexes in the route
        
        dist_table: 2D matrix with the distance between the ith and jth cities
        
    Returns:
        
        distance
    """
        
    distance = 0
            
    city = 0
    
    # The routes only have ncities-1 because the starting one doesn't count
    # since it's always the same (city index 0)
    for next in route:
                    
        distance += dist_table[city][next]
        
        city = next
        
    # Now go back to the starting city
    distance += dist_table[city][0]
    
    return distance

             
def population_fitness(dist_table, population):
    """
    Compute the fitness of each element in the population. The fittest are the
    ones with smallest total distance
    
    Parameters:
    
        dist_table: 2D matrix with the distance between the ith and jth cities
                   
        population: list of routes in the population (note to GA users: routes
                    must be decoded, ie phenotypes!)
        
    Returns:
    
        list with the fitness of each in the range [0,1]
    """    
    
    goals = []
    
    for route in population:
        
        goals.append(total_distance(route, dist_table))
        
    maxdist = max(goals)
    
    mindist = min(goals)
        
    a = -0.8/(maxdist - mindist)
            
    fitness = []
    
    for dist in goals:
        
        fit = 0.6*math.exp(-abs((dist - mindist)/mindist)) + 0.2

#        fit = a*(dist - mindist) + 0.9
#        
#        if fit > 0.9:
#            
#            fit = 0.9
#            
#        if fit < 0.1:
#            
#            fit = 0.1            
        
        fitness.append(fit)
        
    return fitness


# GENETIC ALGORITHM FUNCTIONS
################################################################################
def phenotype(chromosome):
    """
    Decodes the chromosome and returns the city indexes of the route.
    City 0 is always the start of the route.
    
    Parameters:
        
        chromosome: list of reduced indexes.
        
    Returns:
    
        list of city indexes
    """
        
    ncities = len(chromosome) + 1
    
    buffer = range(1, ncities)
    
    route = []
    
    for gene in chromosome:

        city = buffer.pop(gene)
        
        route.append(city)
        
    return route
    
    
def genotype(route):
    """
    Encode the route to obtain its chromosome.
    
    Parameters:
        
        route: list of city indexes describing the route (doesn't include city
               0 because the route always start and end at 0)
        
    Returns:
    
        chromosome as a list of reduced indexes
    """
            
    ncities = len(route) + 1
    
    buffer = range(1, ncities)
    
    chromosome = []
    
    for city in route:
        
        gene = buffer.index(city)

        chromosome.append(gene)
        
        buffer.pop(gene)
        
    return chromosome


def new_generation(population, fitness, crossover_prob):
    """
    Create a new generation by doing crossovers in the old population.
    The crossover happens according to the fitness of each individual and also
    a crossover probability.
    
    Parameters:
    
        population: list of individuals. Each individual is an encoded route
                    (a genotype of the route)
        
        fitness: list with the fitness value of each individual. Values must be
                 in the range [0,1]
                 
        crossover_prob: probability of a crossover happening
        
    Returns:
        
        list with the new population
    """
    
    assert len(population) == len(fitness), \
        "Must have a fitness value for each individual."
        
    assert 1 >= crossover_prob >= 0, \
        "Crossover probability must be with range [0,1]"
    
    popsize = len(population)
    
    new_population = []
    
    for male in xrange(popsize):
        
        for female in xrange(male + 1, popsize):
            
            if hm.utils.flip(crossover_prob) and \
               hm.utils.flip(fitness[male]) and \
               hm.utils.flip(fitness[female]):
                
                child1, child2 = hm.genetic.crossover(population[male], \
                                                      population[female])
                
                # Need to append one child at a time so that the new population
                # doesn't exceed the population size
                new_population.append(child1)
                
                if len(new_population) == popsize:
                    
                    return new_population
                                
                new_population.append(child2)
                
                if len(new_population) == popsize:
                    
                    return new_population
                
    if len(new_population) < popsize:
        
        missing = popsize - len(new_population)
        
        ncities = len(population[0]) + 1
        
        random_guys = hm.randomwalks.salesman(missing, ncities)
        
        encoded_guys = [genotype(route) for route in random_guys]
                
        new_population.extend(encoded_guys)
                
    return new_population


def mutate(individual):
    """
    Mutate (in-place) an individual route by changing the path after a random 
    point.
    
    Parameters:
    
        individual: list with the reduced (encoded) city indexes in the route,
                    ie the genotype
    """
    
    ngenes = len(individual)
    
    index = numpy.random.randint(ngenes)
    
    # This is the size the buffer had when picking then index-th city
    buffersize = ngenes - index
    
    # Pick a new random city
    newgene = individual[index]
    
    # When the buffersize is 1, there is only one choice of gene (0)
    while newgene == individual[index] and buffersize > 1:
        
        newgene = numpy.random.randint(buffersize)
        
    individual[index] = newgene
    
    return index
    
    
def selective_pressure(population, fitness, dist_table, percent):
    """
    Apply some selective pressure to a population. Basically means killing some
    people and putting random individuals in their place.
    
    Parameters:
            
        population: list of individuals. Each individual is an encoded route
                    (a genotype of the route)
        
        fitness: list with the fitness value of each individual. Values must be
                 in the range [0,1]
                    
        dist_table: 2D matrix with the distance between the ith and jth cities
                    
        percent: decimal percentage of individuals to kill        
    
    OBS: replaces the original population
    """
    
    assert 1 >= percent >= 0, \
        "Percentage of survivors must be with range [0,1]"
        
    ncities = len(dist_table)
    
    popsize = len(population)
    
    killsize = int(percent*popsize)
    
    killstart = numpy.random.randint(popsize - killsize)
    
    routes = hm.randomwalks.salesman(killsize, ncities)
    
    replacements = [genotype(route) for route in routes]
    
    population[killstart:killstart + killsize] = replacements
    
    new_fitness = population_fitness(dist_table, routes)
    
    fitness[killstart:killstart + killsize] = new_fitness
    
    
def solve_ga(dist_table, ncities, pop_size, kill_percent=0.3, \
             mutation_prob=0.005, crossover_prob=0.10, max_gen=1000):
    """
    Solve the problem using Genetic Algorithm
    """
    
    # Count for how long the total distance of the generation best hasn't 
    # changed and use this as a stopping criterium
    stagnated_for = 0
    
    # Start with a random population
    phenotypes = hm.randomwalks.salesman(pop_size, ncities)
    
    fitness = population_fitness(dist_table, phenotypes)
    
    population = [genotype(route) for route in phenotypes]
    
    # Record the best
    generation_best = hm.utils.best_by_fitness(phenotypes, fitness)
           
    best_individuals = [generation_best]
    
    distances = [total_distance(generation_best, dist_table)]
    
    best_distances = [distances[-1]]
    
    for generation in xrange(max_gen):
       
        # Need to sort so that the best individuals try to mate first
        hm.utils.sort_by_fitness(population, fitness)
        
        # Mate for a new generation
        population = new_generation(population, fitness, crossover_prob)
        
        # Mutate the unlucky ones
        for individual in population:
            
            if hm.utils.flip(mutation_prob):
                
                mutate(individual)
    
        # Decode the population to recompute their fitness
        phenotypes = [phenotype(chromosome) for chromosome in population]
        
        fitness = population_fitness(dist_table, phenotypes)
        
        # Record the best and check if should stop    
        generation_best = hm.utils.best_by_fitness(phenotypes, fitness)
    
        distances.append(total_distance(generation_best, dist_table))
                   
        if distances[-1] < best_distances[-1]:
               
            best_individuals.append(generation_best)
        
            best_distances.append(distances[-1])
                
        if generation >= 0.5*max_gen and \
           abs((distances[-1] - distances[-2])/distances[-2]) < 10**(-7):
            
            stagnated_for += 1
            
            if stagnated_for == int(0.1*max_gen):
            
                break
        
        # Kill a percentage of the individuals so that the algorithm doesn't
        # stagnate
        selective_pressure(population, fitness, dist_table, kill_percent)
                                            
    return best_individuals, best_distances, distances        