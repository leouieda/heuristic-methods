import numpy

from evolutionary import mean_recomb, wmean_recomb, recomb_pop

x = numpy.arange(0, 10, 1)

y = numpy.arange(1, 11, 1)

print "x:", x
print "y:", y
print "Recombinations:"

z = mean_recomb(x, None, y, None)
print "  mean:"
print "    ", z

x_fit = 3
y_fit = 1
z = wmean_recomb(x, x_fit, y, y_fit) 
print "  weighted mean (x_fit=%g, y_fit=%g):" % (x_fit, y_fit)
print "    ", z

z = range(2, 12)
pop = [x, y, z]
newpop = recomb_pop(pop, [1,1,0], mean_recomb, lower=numpy.zeros(10), \
                    upper=10*numpy.ones(10))
print "  pop:"
for element in pop:
    print "    old:", element

for element in newpop:
    print "    new:", element

