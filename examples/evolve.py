import numpy
import pylab
import math

from evolutionary import evolve
from test_functions import twomin, eggbox

def fitness(pop):
    
    goals = []
    
    for element in pop:
        
        goals.append(twomin(*element))
        
    fit = []
    
    for goal in goals:
        
        fit.append(math.exp(-abs((goal - min(goals))/min(goals))))
        
    return fit

lower = [-10]
upper = [10]
best, best_goal = evolve(fitness, goal=twomin, dims=1, pop_size=100, \
                         lower=lower, upper=upper, prob_mutation=0.005, \
                         mutation_size=0.2)

print "Best goal:", min(best_goal)
print "Best:", best[best_goal.index(min(best_goal))][0]

pylab.figure()
pylab.title("TwoMin")
x = numpy.arange(lower[0], upper[0], 0.5)
y = twomin(x)

pylab.plot(x, y, '-b')

best_x = [x[0] for x in best]

pylab.plot(best_x, best_goal, '*k')

pylab.show()