"""
Number encoding and decoding.
"""

import math
import numpy


def bin2int(binary):
    """
    Convert a binary sequence to an unsigned integer.
    
    Parameters:
    
        binary: list with the binary sequence
        
    Returns: integer
    """
    
    integer = 0
    
    nbits = len(binary)
    
    for i in xrange(nbits):
        
        integer += binary[i]*(2**i)
        
    return integer
    

def int2bin(integer, nbits):
    """
    Return the binary representation of integer as a list of ones and zeros.
    """
    
    assert type(integer) is int and type(nbits) is int
    
    tmp = integer
    
    binary = numpy.zeros(nbits, dtype='int')
    
    for i in xrange(nbits):
        
        bit = tmp%2
        
        binary[i] = bit
        
        tmp /= 2
        
        if tmp == 0:
        
            break
    
    return binary


def encode(number, lower, upper, sig_digits):
    """
    Convert a floating point number to binary representation.
    
    Parameters:
    
        number: float to be converted
        
        lower: lower range allowed for 'number'
        
        upper: upper range allowed for 'number'
        
        sig_digits: number of significant digits in the representation
    """

    nbits = int(math.ceil(sig_digits*10./3.))
    
    const = float(number - lower)/(upper - lower)
    
    integer = int((2**nbits - 1)*const)

    return int2bin(integer, nbits)

    
def decode(binary, lower, upper):
    """
    Decode a binary sequence into a floating point in the range [lower,upper]
    """

    nbits = len(binary)
    
    constant = float(upper - lower)/(2**nbits - 1)
    
    return constant*bin2int(binary) + lower



if __name__ == '__main__':
    
    number = 0.1
    lower = -10**2
    upper = 10**2
    sig_dig = 5
    
    print "Encoding %g to binary:" % (number)
    print "  range = [%g, %g]" % (lower, upper)
    print "  %d significant digits" % (sig_dig)
    a = encode(number, lower, upper, sig_dig)
    print a
    print "Used %d bits." % (len(a))
    print "Decoding:"
    b = decode(a, lower, upper)
    
    print '%.15f' % (b)