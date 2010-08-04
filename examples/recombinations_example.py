import numpy

from evolutionary import mean_recomb, wmean_recomb, recomb_pop

x = numpy.arange(0, 10, 1)

y = numpy.arange(1, 11, 1)

print "x:", x
print "y:", y
print "Recombinations:"

z = mean_recomb(x, y)
print "  mean:"
print "    ", z

x_fit = 3
y_fit = 1
z = wmean_recomb(x, x_fit, y, y_fit) 
print "  weighted mean (x_fit=%g, y_fit=%g):" % (x_fit, y_fit)
print "    ", z


def fitness(estimate):
    return 0

pop = [x, y]

newpop = recomb_pop(pop, fitness, mean_recomb, lower=numpy.zeros(10), \
                    upper=10*numpy.ones(10))

print newpop

