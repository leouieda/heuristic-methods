"""
Run the Genetic Algorithm in the eggbox function
"""

import math

import numpy
import pylab

import hm.functions
import hm.genetic

    
print "Optimizing the Eggbox function"

lower = [-20,-20]
upper = [20,20]
step = 0.5 # Discretization size to make the plot of the eggbox
sig_digits = [5]*2
pop_size = 200
mutation_prob = 0.01
crossover_prob = 0.7
max_it = 100

best, best_goal, goals = hm.genetic.solve(hm.functions.eggbox, \
                                          pop_size, sig_digits, \
                                          lower, upper, \
                                          mutation_prob, crossover_prob, \
                                          max_it)

print "Best solution found:", best[-1]

best_x, best_y = numpy.array(best).T

# Grid and plot the eggbox
xs = numpy.arange(lower[0], upper[0] + step, step)
ys = numpy.arange(lower[1], upper[1] + step, step)
gridx, gridy = pylab.meshgrid(xs, ys)

gridegg = hm.functions.eggbox(gridx, gridy)

pylab.figure()
pylab.title("Eggbox")
pylab.axis('scaled')

pylab.pcolor(gridx, gridy, gridegg, cmap=pylab.cm.jet)

pylab.plot(best_x, best_y, '.-k', label="Best estimates", markersize=8)
pylab.plot(best_x[-1], best_y[-1], '^g', label="Best ever", markersize=8)

pylab.legend(prop={'size':10})

pylab.xlim(lower[0], upper[0])
pylab.ylim(lower[1], upper[1])

pylab.figure()
pylab.suptitle("Eggbox per generation")

# Plot the advance of the generation best
pylab.subplot(2,1,1)
pylab.plot(goals, '.-k', label="Generation best")
pylab.ylabel("Eggbox function")
pylab.legend(prop={'size':10})

# Plot the advance of the best solution found
pylab.subplot(2,1,2)
pylab.plot(best_goal, '.-k', label="Best ever")
pylab.xlabel("Iteration")
pylab.ylabel("Eggbox function")
pylab.legend(prop={'size':10})

pylab.show()