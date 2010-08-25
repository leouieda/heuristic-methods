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
        
        if child1[i] != child2[i]:
            
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
        
        nbits = len(population[0])
        
        random_guys = hm.randomwalks.genetic(missing, nbits)
                
        new_population.extend(random_guys)
                
    return new_population


def phenotype(chromosome, bits_per_dim, lower, upper, fmt='gray'):
    """
    Decode the whole chromosome to produce a human-readable list of float 
    variables.
    
    Parameters:
    
        chromosome: lits of binary values describing the chromosome.
        
        bits_per_dim: list with the number of bits used for each variable
        
        lower: list of lower bounds of the parameter space
        
        upper: list of upper bounds of the parameter space
        
    Returns:
        list of decoded variables.
    """
    
    assert len(bits_per_dim) == len(lower) == len(upper), \
        "bits_per_dim, lower and upper must have the same length"
    
    ndims = len(bits_per_dim)
    
    res = []
    
    start = 0
    
    for dim, nbits in zip(range(ndims), bits_per_dim):
        
        end = start + nbits
        
        var = hm.codec.decode(chromosome[start:end], lower[dim], upper[dim], \
                              fmt)
        
        res.append(var)
                
        start = end
        
    return res

              
def population_fitness(function, population):
    """
    Compute the fitness of each element in the population. The fittest are the
    ones with smallest value of 'function'
    
    Parameters:
    
        function: function object that evaluates the goal function of each 
                  element. Elements are lists of parameters.
                  
        population: list of elements with the PHENOTYPES of the population
        
    Returns:
    
        list with the fitness of each in the range [0,1]
    """    
    
    goals = []
    
    for element in population:
        
        goals.append(function(*element))
        
    fit = []
    
    for goal in goals:
        
        fit.append(math.exp(-abs((goal - min(goals))/min(goals))))
        
    return fit    
              
              
def solve(function, pop_size, sig_digits, lower, upper, \
          mutation_prob=0.005, crossover_prob=0.10, max_it=1000):
    """
    Genetic Algorithm.
    Evolves a population to find the best solution according to a fitness
    criterium.
    """
    
    bits_per_dim = [int(math.ceil(n*10./3.)) for n in sig_digits]
    
    totalbits = sum(bits_per_dim)
    
    population = hm.randomwalks.genetic(pop_size, totalbits)
    
    float_pop = [phenotype(chromo, bits_per_dim, lower, upper) \
                 for chromo in population]
    
    fitness = population_fitness(function, float_pop)
       
    hm.utils.sort_by_fitness(population, fitness)
           
    generation_best = phenotype(population[0], bits_per_dim, lower, upper)
           
    best_individuals = [generation_best]
    
    goals = [function(*generation_best)]
    
    best_goals = [goals[-1]]
        
    for generation in xrange(max_it):
        
        # Mate for a new generation
        population = new_generation(population, fitness, crossover_prob)
        
        # Mutate the unlucky ones
        for i in xrange(pop_size):
            
            if hm.utils.flip(mutation_prob):
                
                mutate(population[i])           
    
        # Decode the population to recompute their fitness
        float_pop = [phenotype(chromo, bits_per_dim, lower, upper) \
                     for chromo in population]
        
        fitness = population_fitness(function, float_pop)
       
        hm.utils.sort_by_fitness(population, fitness)
                      
        generation_best = phenotype(population[0], bits_per_dim, lower, upper)
        
        goals.append(function(*generation_best))
        
        if goals[-1] < best_goals[-1]:
               
            best_individuals.append(generation_best)
        
            best_goals.append(goals[-1])
                
        if generation >= 0.5*max_it and \
           abs((goals[-1] - goals[-2])/goals[-2]) < 10**(-7):
            
            break        
                                
    return best_individuals, best_goals, goals
                
                