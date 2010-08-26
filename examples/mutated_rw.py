import numpy
import pylab

from montecarlo import mutated_rw
from test_functions import twomin

lower = -10
upper = 10
x = numpy.arange(lower, upper, 0.1)
y = twomin(x)

print "Mutated Random Walk:"

best_param, best_goal, params, goals = mutated_rw(func=twomin, \
                                                  lower=lower, upper=upper, \
                                                  num_agentes=10, \
                                                  prob_mutation=0.05, \
                                                  size_mutation=0.2, \
                                                  max_it=1000, threshold=0.9)

print "Best: %g at %g" % (best_goal, best_param)

pylab.plot(x, y, '-k')

pylab.plot(params, goals, '.k')

pylab.show()