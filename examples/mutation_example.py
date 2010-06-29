import pylab

from montecarlo import mutation

value = 9
lower = 0
upper = 10
size = 0.2
n=10000

mut = [mutation(value, lower, upper, size) for i in xrange(n)]

pylab.figure()
pylab.hist(mut, bins=int(0.1*n), facecolor='gray')
pylab.show()
