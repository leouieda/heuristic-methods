"""
Find the minimum of the twomin function with Newton's method
"""

import pylab
import numpy

import hm.gradient
import hm.functions

# TWOMIN
x = numpy.arange(-10, 10.1, 0.1)
y = hm.functions.twomin(x)

print "Solving..."

result = hm.gradient.newton(func=hm.functions.twomin, dims=1, \
                            initial=[9], maxit=100, stop=10**(-15))

solution, goal, estimates, goals = result

print "Minimum found:", goal, "at", solution[0]
print "Took %d iterations" % (len(goals))

pylab.figure(figsize=(15,6))
pylab.suptitle("Newton's method of optimization")

pylab.subplot(1,2,1)
pylab.title("Function value per iteration")
pylab.plot(goals, '.-k')
pylab.xlabel("Iteration")

pylab.subplot(1,2,2)
pylab.title("Twomin function")
pylab.plot(x, y, '-b')
pylab.plot(estimates, goals, '*-k', label='Steps')
pylab.plot(solution, goal, '^y', markersize=8, label="Final solution")
pylab.xlabel("x")

pylab.legend(prop={'size':9})

pylab.show()