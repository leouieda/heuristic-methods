"""
Collection of gradient optimizers:
    1) Newton's method
"""

import numpy


def gradient(func, dims, params, delta):
    """
    Calculate the gradient of func evaluated at params
    """    
        
    grad = numpy.zeros(dims)
    
    tmp = numpy.zeros(dims)
        
    # Compute the gradient
    for i in xrange(dims):
        
        tmp[i] = delta
        
        grad[i] = (func(*(params + tmp)) - func(*(params - tmp)))/delta
        
        tmp[i] = 0

    return grad


def hessian(func, dims, params, delta):
    """
    Calculate the Hessian matrix of func evaluated at params
    """

    hessian = numpy.zeros((dims, dims))
    
    tmpi = numpy.zeros(dims)
    
    tmpj = numpy.zeros(dims)
    
    for i in xrange(dims):
    
        tmpi[i] = delta
        
        params1 = params + tmpi
        
        params2 = params - tmpi    
        
        for j in xrange(i, dims):
        
            tmpj[j] = delta
            
            deriv2 = (func(*(params2 + tmpj)) - func(*(params1 + tmpj)))/delta
            
            deriv1 = (func(*(params2 - tmpj)) - func(*(params1 - tmpj)))/delta
            
            hessian[i][j] = (deriv2 - deriv1)/delta
            
            # Since the Hessian is symmetric, spare me some calculations
            hessian[j][i] = hessian[i][j]
            
            tmpj[j] = 0
        
        tmpi[i] = 0
    
    return hessian
    
    
def newton(func, dims, initial, maxit=100, stop=10**(-15)):
    """
    Newton's method of optimization.
    Note: the derivative of func is calculated numerically
    """
    
    delta = 0.0001
    
    solution = numpy.copy(initial)
    
    estimates = [solution]
    
    goals = [func(*solution)]
    
    for i in xrange(maxit):
        
        grad = gradient(func, dims, solution, delta)
        
        H = hessian(func, dims, solution, delta)
        
        correction = numpy.linalg.solve(H, grad)
        
        solution = solution + correction
        
        estimates.append(solution)
        
        goals.append(func(*solution))
        
        if abs((goals[i] - goals[i - 1])/goals[i - 1]) <= stop:
            
            break
        
    return solution, goals[-1], numpy.array(estimates), goals
