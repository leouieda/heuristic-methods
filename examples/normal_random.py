from montecarlo import normal_rand

import pylab

num = 10000

values = [normal_rand(mean=0, stddev=10) for i in xrange(num)]

pylab.figure()

pylab.hist(values, bins=int(0.1*num), facecolor='gray')

pylab.show()