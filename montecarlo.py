"""
Monte Carlo optimizers.
"""

import numpy


def twomin(x):
    
    return (x + 0.5)**4 - 30*(x**2) - 20*x


def flip(bias):
    """
    Flips a biased coin.
    Returns 1 for heads, 0 for tails 
    
    Parameters:
    
        bias: probability of returning heads
    """
    
    res = numpy.random.random()
        
    if res > bias:
        
        return 0
    
    else:
        
        return 1
    
    

# TODO: allow for different lower and upper for each dimension 
def randwalk(func, dims, randnum, lower, upper):
    """
    Optimize using a random walk.
    
    Parameters:
        
        func: function object to be optimized
        
        dims: number of arguments the function receives
        
        randnum: number of random points to generate
        
        lower: lower bound of the parameter space
        
        upper: upper bound of the parameter space
        
    Returns:
    
        [values, parameters, 
    """
    
    estimates = []
    
    values = []
    
    for i in xrange(randnum):
                
        estimate = numpy.random.uniform(lower, upper, dims)
        
        value = func(*estimate)
        
        if i != 0:
        
            if value < values[-1]:            
        
                estimates.append(estimate)
    
                values.append(value)
                
            elif flip(0.20):     
        
                estimates.append(estimate)
    
                values.append(value)
    
        else:     
        
            estimates.append(estimate)
    
            values.append(value)
            
    # Sort the values and estimates
    indexes = numpy.argsort(values)

    sorted_vals = []
    
    sorted_params = []
    
    for i in xrange(len(values)):
        
        sorted_vals.append(values[indexes[i]])
    
        sorted_params.append(estimates[indexes[i]].tolist())
                
    return sorted_vals, sorted_params


if __name__ == '__main__':
    
    vals, params = randwalk(func=twomin, dims=1, \
                            randnum=10, lower=-10, upper=10)
    
    print "Max:"    
    print vals[0]
    print "Parameter:"
    print params[0]
    
    
    