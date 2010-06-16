"""
Collection of gradient optimizers
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
        
        if abs(goals[i] - goals[i - 1])/goals[i - 1] <= stop:
            
            break
        
    return solution, goals[-1], numpy.array(estimates), goals


if __name__ == '__main__':
    
    from test_functions import twomin, eggbox
    import pylab
    
    # TWOMIN
    x = numpy.arange(-10,10)
    y = twomin(x)
    
    pylab.figure()
    pylab.title("Twomin")
    pylab.plot(x, y, '-b')
    
    solution, goal, estimates, goals = newton(func=twomin, dims=1, \
                                initial=[10], maxit=100, stop=10**(-15))
    
    print "Min =", goal, "at", solution
    print "Took %d iterations" % (len(goals))

    pylab.plot(estimates, goals, '*k')
    pylab.plot(solution, goal, '^y')
    
    pylab.figure()
    pylab.title("Twomin Goal function")
    pylab.plot(goals, '-k')  
    
    # EGGBOX
    x = numpy.arange(-25, 25.5, 0.5)
    y = numpy.arange(-25, 25.5, 0.5)    
    X, Y = pylab.meshgrid(x, y)
    Z = eggbox(X, Y)
    
    best_estimate, best_goal, estimates, goals = newton(func=eggbox, dims=2, \
                        initial=[11.6,0.7], maxit=100, stop=10**(-15))
        
    print "Best solution:", best_goal
    print "at:", best_estimate
        
    fig = pylab.figure()
    
    pylab.title("Eggbox")
    pylab.contourf(X, Y, Z, 40)
    pylab.colorbar()
    
    pylab.plot(estimates.T[0], estimates.T[1], '*k')   
    pylab.plot(0, 0, 'oy', label='global minimum')    
    pylab.plot(best_estimate[0], best_estimate[1], '^c', label='best solution') 
    pylab.legend(numpoints=1, prop={'size':9})
        
    pylab.xlabel("X")
    pylab.ylabel("Y")
    pylab.xlim(-25, 25)
    pylab.ylim(-25, 25)
    
    pylab.figure()
    pylab.title("Eggbox Goal function")
    pylab.plot(goals, '-k')
    pylab.xlabel("Iteration")
    
    pylab.show()