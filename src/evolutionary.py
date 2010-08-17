"""
Evolutionary optimizations
"""


import numpy

from hm_random import flip


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


def mean_recomb(x, x_fit, y, y_fit):
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



def recomb_pop(population, pop_fitness, recomb, lower, upper):
    """
    Recombines the whole population according to a fitness function and a 
    recombination strategy.
    
    Parameters:
    
        population: list of elements in the population. Each element should be
            an Nx1 dimensional array like
            
        pop_fitness: list of the fitness of each element in the population
            
        recomb: function object of the recombination strategy
        
        lower: list with the lower bounds of the parameter space
        
        upper: list with the upper bounds of the parameter space
    """
    
    assert len(lower) == len(upper) == len(population[0]), \
        "upper and lower must have the same number of elements " + \
        "as there are dimensions in the problem"

    assert len(pop_fitness) == len(population), \
        "Need a fitness value for every element in the population."

    new_pop = []
    
    pop_size = len(population) 
    
    for i, element, fitness in zip(xrange(pop_size), population, pop_fitness):
                
        if flip(fitness):
            
            left_over_pop = population[0:i] + population[i+1:]
            left_over_fit = pop_fitness[0:i] + pop_fitness[i+1:]
            
            for mate, mate_fitness in zip(left_over_pop, left_over_fit):
                
                if flip(mate_fitness):
                    
                    child = recomb(element, fitness, mate, mate_fitness)

                    new_pop.append(child)
                    
                    if len(new_pop) == pop_size:
                        
                        break
        
        if len(new_pop) == pop_size:
            
            break
    
    while len(new_pop) != pop_size:
        
        estimate = []
        
        for l, u in zip(lower, upper):
            
            estimate.append(numpy.random.uniform(l, u))
            
        new_pop.append(numpy.array(estimate))
    
    return new_pop
            


def evolve(fitness, goal, dims, pop_size, lower, upper, prob_mutation=0.005, \
           mutation_size=0.1, max_it=100):
    """
    Evolve a population using a given fitness criterion.
    
    Parameters:
    
        fitness: function object that calculates the fitness of each element in
                 the population. Input: list of elements representing the 
                 population. Each element is a list of size 'dims'. 
                 Output: List with the fitness of each element as a float value
                 in the range [0:1]
                 
        goal: function object that calculates the goal function due to an
              element of the population
                 
        dims: number of dimensions in the problem
        
        pop_size: how many individuals in the population
        
        lower: list with the lower bounds of the parameter space
        
        upper: list with the upper bounds of the parameter space
        
        prob_mutation: the probability of a mutation occurring at each iteration
        
        mutation_size: percentage of the parameter space that will be allowed
                       during the mutation
        
        max_it: maximum number of iterations
        
    Output:
        
        [best_elements, best_goals]
            best_elements: list with the optimum element at each iteration
            best_goals: list with the goal function value of each optimum 
                        element       
    """
    
    assert len(lower) == len(upper) == dims, \
        "upper and lower must have the same number of elements " + \
        "as there are dimensions in the problem"
    
    assert prob_mutation >= 0 and prob_mutation <= 1, \
        "prob_mutation must be in the range [0:1]"
    
    
    # Begin with a random population
    population = []
    
    for i in xrange(pop_size):
                
        estimate = []
        
        for l, u in zip(lower, upper):
            
            estimate.append(numpy.random.uniform(l, u))
            
        population.append(numpy.array(estimate))
        
    pop_fitness = fitness(population)    
    
    best_fit = max(pop_fitness)
    
    best = [population[pop_fitness.index(best_fit)]]
    
    best_goal = [goal(*best[-1])]
    
    # Evolve until reaching stagnation or max_it
    for iteration in xrange(max_it):
        
        population = recomb_pop(population, pop_fitness, wmean_recomb, \
                                lower, upper)
    
        # Mutate the unlucky ones
        for element in population:
            
            if flip(prob_mutation):
                
                p = numpy.random.randint(dims)
                
                element[p] = mutate(element[p], lower[p], upper[p], \
                                    size=mutation_size)
        
        pop_fitness = fitness(population)
    
        best_fit = max(pop_fitness)
        
        best.append(population[pop_fitness.index(best_fit)])
            
        best_goal.append(goal(*best[-1]))
        
        print "it %d:" % (iteration)
        print "  best goal:", best_goal[-1]
        print "  best:", best[-1]

        if abs((best_goal[-1] - best_goal[-2])/best_goal[-2]) <= 10**(-4):
            
            break
    
    return best, best_goal
    
    
    
def discrete_crossover(male, female):
    """
    Cross the genes of the 2 mates given. (Overwrites the arrays)
    """
    
    assert len(male) == len(female), \
        "Male and female must have the same number of genes"
    
    ngenes = len(male)
        
    cross_point = numpy.random.randint(ngenes)
    
    if cross_point > 0.5*ngenes:
        
        rest = range(cross_point, ngenes) 
    
    else:
        
        rest = range(0, cross_point)
        
    for i in rest:
    
        tmp = male[i]
        male[i] = female[i]
        female[i] = tmp
        
        
def dicrete_mutate(individual):
    """
    Swap two random genes of individual
    """
    
    ngenes = len(individual)
    
    gene1 = numpy.random.randint(ngenes)
    
    gene2 = numpy.random.randint(ngenes)
    
    tmp = individual[gene1]
    
    individual[gene1] = individual[gene2]
    
    individual[gene2] = tmp
        