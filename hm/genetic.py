"""
Genetic algorithm solver
"""

import numpy
import math

import hm.utils
import hm.codec
import hm.randomwalks


def crossover(male, female):
    """
    Do a crossover between 2 individuals.
    
    Parameters:
    
        male: list with binary values describing the chromosome of the first
              mate
              
        female: list with binary values describing the chromosome of the second
              mate
              
    Returns:
    
        list with the 2 children
              
    OBS: Order of male and female doesn't matter.
    """
    
    assert len(male) == len(female), \
        "Male and female must have same number of bits (elements)"
    
    nbits = len(male)
    
    child1 = numpy.copy(male)
    
    child2 = numpy.copy(female)
    
    middle = numpy.random.randint(nbits)
    
    for i in xrange(middle, nbits):
        
        tmp = child1[i]
        
        child1[i] = child2[i]
        
        child2[i] = tmp
        
    return child1, child2


def mutate(individual):
    """
    Flip a random bit in individual.
    
    Parameters:
    
        individual: list of binary values describing the individuals chromosome
    """
    
    bit = numpy.random.randint(len(individual))
    
    if individual[bit] == 0:
        
        individual[bit] = 1
    
    else:
        
        individual[bit] = 0
        
        
def new_generation(population, fitness, crossover_prob):
    """
    Create a new generation by doing crossovers in the old population.
    The crossover happens according to the fitness of each individual and also
    a crossover probability.
    
    Parameters:
    
        population: list of individuals. Each individual is a list of bits
        
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
                
                child1, child2 = crossover(population[male], population[female])
                
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
        
        nbits = len(new_population[0])
        
        random_guys = hm.randomwalks.genetic(missing, nbits)
                
        new_population.extend(random_guys)
                
    return new_population
                
              
def ga(pop_size, sig_digits, lower, upper, fitness_func, \
       mutation_prob=0.005, crossover_prob=0.10, max_it=1000):
    """
    Genetic Algorithm.
    Evolves a population to find the best solution according to a fitness
    criterium.
    """
    
    bits_per_dim = [int(math.ceil(n*10./3.)) for n in sig_digits]
    
    totalbits = sum(bits_per_dim)
    
    population = hm.randomwalks.genetic(pop_size, totalbits)
    
    fitness = fitness_func(population)
       
    hm.utils.sort_by_fitness(population, fitness)
           
    best_individual = numpy.copy(population[0])
    
    best_fitness = fitness[0]
    
    for generation in xrange(max_it):
        
        population = new_generation(population, fitness, crossover_prob)
        
        for i in xrange(pop_size):
            
            if hm.utils.flip(mutation_prob):
                
                mutate(population[i])
                
        fitness = fitness_func(population)
       
        hm.utils.sort_by_fitness(population, fitness)
        
        if fitness[0] > best_fitness:
               
            best_individual = numpy.copy(population[0])
            
            best_fitness = fitness[0]
            
    best_individual = hm.codec.decode(best_individual, lower, upper)
            
    return best_individual, best_fitness
                
                