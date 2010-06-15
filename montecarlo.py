"""
Collection of Monte Carlo optimizers 
"""

import numpy

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
        

def randwalk(func, dims, lower, upper, steps, maxit, threshold=0.40):
    """
    Optimize using a random walk.
    
    Parameters:
        
        func: function object to be optimized
        
        dims: number of arguments the function receives
        
        lower: array with lower bounds of the parameter space
        
        upper: array with upper bounds of the parameter space
        
        steps: step size in each dimension
        
        maxit: maximum iterations
        
        threshold: chance of accepting an upward step (range [0:1])
        
    Returns:
    
        [best_goal, best_estimate, goals, estimates]:
        
            best_goal: the smallest value of the goal function that was found
            
            best_estimate: point in the parameter space where 'best_goal' was 
                           found
                           
            goals: list of the goal function through the iterations
            
            estimates: list of the points where 'goals' where found  
    """
    
    max_inner = 20
    
    # Make an initial estimate
    estimate = []
    
    for i in xrange(dims):
        
        estimate.append(numpy.random.uniform(lower[i], upper[i]))
        
    estimate = numpy.array(estimate)
    
    # Keep all the steps recorded
    estimates = [estimate]
    
    goals = [func(*estimate)]    
                
    # Start walking on the parameter space    
    step = numpy.empty_like(estimate)

    for iteration in xrange(maxit):
        
        for i in xrange(dims):
            
            step[i] = numpy.random.uniform(-steps[i], steps[i])
            
        tmp = estimate + step
        
        goal = func(*tmp)
        
        inner_it = 0
        
        while goal >= goals[-1]:
        
            for i in xrange(dims):
            
                step[i] = numpy.random.uniform(-steps[i], steps[i])
            
            tmp = estimate + step
        
            goal = func(*tmp)
                 
            if goal >= goals[-1] and flip(threshold) or inner_it >= max_inner:
                
                if inner_it >= max_inner:
                    
                    step = numpy.zeros_like(step)
                    
                break
            
            inner_it += 1
            
        estimate = estimate + step
        
        # Mark the best estimate so far
        if iteration != 0:
            
            if goal < best_goal:
                
                best_goal = goal
                
                best_estimate = numpy.copy(estimate)
        
        else:
            
            best_goal = goal
            
            best_estimate = numpy.copy(estimate)         
        
        estimates.append(estimate)

        goals.append(goal)     
        
    return best_goal, best_estimate, goals, numpy.array(estimates)



if __name__ == '__main__':
        
    import pylab
    from test_functions import eggbox
    
    print "Minimizing an eggbox function using random walk" 
    
    x = numpy.arange(-25, 25.5, 0.5)
    y = numpy.arange(-25, 25.5, 0.5)    
    X, Y = pylab.meshgrid(x, y)
    Z = eggbox(X, Y)
    
    best_goal, best_estimate, goals, estimates = randwalk(func=eggbox, dims=2, \
                        lower=(-20,-20), upper=(20,20), \
                        steps=(10,10), maxit=500, threshold=0.4)
        
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
    