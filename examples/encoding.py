"""
Encode and decode floating point numbers to binary format
"""

import hm.codec
    
number = 0.0
lower = -8
upper = 8
sig_dig = 5

print "Encoding %g to binary:" % (number)
print "  range = [%g, %g]" % (lower, upper)
print "  %d significant digits" % (sig_dig)
a = hm.codec.encode(number, lower, upper, sig_dig)
print "Binary:", a
print "Used %d bits." % (len(a))
print "Decoding:"
b = hm.codec.decode(a, lower, upper)

print '%.15f' % (b)