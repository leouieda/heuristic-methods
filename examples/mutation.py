import pylab

from montecarlo import mutate

value = 1
lower = 0
upper = 10
size = 0.5
n=10000

mut = [mutate(value, lower, upper, size) for i in xrange(n)]

pylab.figure()
pylab.hist(mut, bins=int(0.1*n), facecolor='gray')
pylab.show()
