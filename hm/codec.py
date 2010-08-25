"""
Number encoding and decoding.
"""

import math
import numpy


formats = ['gray', 'std']


def gray2int(gray_binary):
    """
    Convert a Gray format binary to an integer.
    
    Parameters:
    
        gray_binary: list of ones and zeros in Gray format
        
    Returns:
        
        Corresponding integer value
    """
    
    integer = 0
    aux = 0
    nbits = len(gray_binary)
    
    for i in xrange(nbits - 1, -1, -1):
        
        if gray_binary[i] == 1:
            
            if aux == 0:
               
                aux = 1
               
            else:
                
                aux = 0
                
        if aux == 1:
            
            integer += 2**i
            
    return integer
         

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

    
def decode(binary, lower, upper, fmt='gray'):
    """
    Decode a binary sequence into a floating point in the range [lower,upper]
    
    Parameters:
    
        binary: list of ones and zeros with the binary code
        
        lower: lower range allowed for 'number'
        
        upper: upper range allowed for 'number'
        
        fmt: either 'gray' for Grays binary format or 'std' for the standard
    """
        
    assert fmt in formats, "Unknown binary format '%s'" % (fmt)

    nbits = len(binary)
    
    constant = float(upper - lower)/(2**nbits - 1)
    
    if fmt == 'gray':
        
        integer = gray2int(binary)
        
    if fmt == 'std':
        
        integer = bin2int(binary)
    
    return constant*integer + lower