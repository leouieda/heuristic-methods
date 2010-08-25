"""
Solve a sample Traveling Salesman problem
"""

import numpy
import pylab

import hm.salesman

cities = [(0,0), (1,0), (0,1), (1,1)]

dist_table = hm.salesman.build_distance_table(cities, type='cartesian')

best, best_dists, dists = hm.salesman.solve_ga(dist_table, ncities=4, \
                                               pop_size=100, \
                                               mutation_prob=0.005, \
                                               crossover_prob=0.7, \
                                               max_it=100)

print best[-1]
print best_dists[-1]

pylab.figure()
pylab.suptitle("Total distance per generation")

# Plot the advance of the generation best
pylab.subplot(2,1,1)
pylab.plot(dists, '.-k', label="Generation best")
pylab.ylabel("Distance")
pylab.legend(prop={'size':10})

# Plot the advance of the best solution found
pylab.subplot(2,1,2)
pylab.plot(best_dists, '.-k', label="Best ever")
pylab.xlabel("Iteration")
pylab.ylabel("Distance")
pylab.legend(prop={'size':10})

pylab.show()