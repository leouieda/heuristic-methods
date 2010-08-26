"""
Solve a sample Traveling Salesman problem
"""

import numpy
import pylab

import hm.salesman

cities = [(0,0), (1,0), (0,1), (1,1), (0.5,1.5), (.5, .8)]
ncities = len(cities)

print "Building the distances table"

dist_table = hm.salesman.build_distance_table(cities, type='cartesian')

print "Solving..."

result = hm.salesman.solve_ga(dist_table, ncities, pop_size=20, \
                              mutation_prob=0.05, crossover_prob=0.7, \
                              max_it=100)

best_routes, best_dists, dists = result

print "Done!"

print "Best total distance:", best_dists[-1]

# PLOTS OF THE DISTANCE PER GENERATION
pylab.figure(figsize=(13,6))
pylab.suptitle("Traveling Salesman: Genetic Algorithm")

pylab.subplot(2,2,1)
pylab.plot(dists, '.-k', label="Generation best")
pylab.ylabel("Distance")
pylab.legend(prop={'size':10})

pylab.subplot(2,2,3)
pylab.plot(best_dists, '.-k', label="Best ever")
pylab.xlabel("Iteration")
pylab.ylabel("Distance")
pylab.legend(prop={'size':10})

# PLOT THE BEST ROUTE FOUND
pylab.subplot(1,2,2)
pylab.axis('scaled')
pylab.title("Best Route: Distance = %g" % (best_dists[-1]))

city = 0
for next in best_routes[-1]:
    
    path_x = [cities[city][0], cities[next][0]]
    path_y = [cities[city][1], cities[next][1]]
    
    pylab.plot(path_x, path_y, 'o-k')

    city = next
    
path_x = [cities[city][0], cities[0][0]]
path_y = [cities[city][1], cities[0][1]]

pylab.plot(path_x, path_y, 'o-k')
pylab.plot(cities[0][0], cities[0][1], 'or', \
           label='Starting point')
pylab.legend(prop={'size':10, }, numpoints=1)

pylab.xlim(-1, 2)
pylab.ylim(-1, 2)

pylab.show()