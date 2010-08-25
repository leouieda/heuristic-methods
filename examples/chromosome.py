"""
Create binary chromosomes and decode them back to floats
"""

import numpy

import hm.codec
import hm.genetic

vars = [3.4, -5.6, 21.78, 3.9]
significant = [10, 5, 8, 6]
lower = [-100]*4
upper = [100]*4

chromo = []
bits = []

for i in xrange(len(vars)):
    
    gene = hm.codec.encode(vars[i], lower[i], upper[i], significant[i])
    bits.append(len(gene))
    chromo.extend(gene)
    
print "Floats:\n ", vars
print "Chromosome (standard binary):\n ", chromo

decvars = hm.genetic.phenotype(chromo, bits, lower, upper, fmt='std')
print "Decoded:\n ", decvars
