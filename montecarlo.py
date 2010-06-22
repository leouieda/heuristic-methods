"""
Collection of Monte Carlo optimizers 
"""

import math

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
        

def randwalk(func, dims, lower, upper, step_sizes, maxit, threshold=0.40):
    """
    Optimize using a random walk.
    
    Parameters:
        
        func: function object to be optimized
        
        dims: number of arguments the function receives
        
        lower: array with lower bounds of the parameter space
        
        upper: array with upper bounds of the parameter space
        
        step_sizes: step size in each dimension
        
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
            
            step[i] = numpy.random.uniform(-step_sizes[i], step_sizes[i])
            
        tmp = estimate + step
        
        goal = func(*tmp)
        
        inner_it = 0
        
        while goal >= goals[-1]:
        
            for i in xrange(dims):
            
                step[i] = numpy.random.uniform(-step_sizes[i], step_sizes[i])
            
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


def simulated_annealing(func, dims, lower, upper, step_sizes, \
                        tstart, tfinal, tstep, it_per_t):
    """
    Optimize using Simulated Annealing.
    
    Parameters:
        
        func: function object to be optimized
        
        dims: number of arguments the function receives
        
        lower: array with lower bounds of the parameter space
        
        upper: array with upper bounds of the parameter space
        
        step: list with the step sizes in each dimension
        
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
    
    # Make random initial estimate
    estimate = []
    
    for i in xrange(dims):
        
        estimate.append(numpy.random.uniform(lower[i], upper[i]))
        
    estimate = numpy.array(estimate)
    
    goal = func(*estimate)
     
    best_estimate = estimate
    
    best_goal = goal 
    
    # Keep all the steps recorded
    estimates = [estimate]
    
    goals = [goal]
        
    step = numpy.empty_like(estimate)    

    for temperature in numpy.arange(tstart, tfinal - tstep, -tstep, dtype='f'):
                
        iteration = 0
        
        while iteration < it_per_t:
            
            # Make a random perturbation
            for i in xrange(dims):
                
                step[i] = numpy.random.uniform(-step_sizes[i], step_sizes[i])
                    
            tmp = estimate + step
            
            goal = func(*tmp)
            
            delta_goal = goals[-1] - goal
            
            survival_prob = math.exp(delta_goal/temperature)
            
            if delta_goal > 0 or flip(survival_prob):
            
                msg = "Temp: %g  iterations: %d  goal: %g" \
                    % (temperature, iteration + 1, goal)
                    
                if delta_goal <= 0:
                    
                    msg += "  survival: %g" % (survival_prob)

                print msg
                            
                estimate = tmp
                
                if goal < best_goal:
                    
                    best_goal = goal
                    
                    best_estimate = estimate        
                
                estimates.append(estimate)
                
                goals.append(goal)              
    
                break
            
            iteration += 1
        
#        if iteration == it_per_t:
#            
#            print "Max iterations reached and couldn't decrease goal"
            
#            break
        
    return best_goal, best_estimate, goals, numpy.array(estimates)     
    
    
    
if __name__ == '__main__':
        
    import pylab
    from test_functions import eggbox
    
    bot = -20
    top = 20
    step = (top - bot)/(50. - 1.)
    step_s = (5,5)
        
    x = numpy.arange(bot, top + step, step)
    y = numpy.arange(bot, top + step, step)    
    X, Y = pylab.meshgrid(x, y)
    Z = eggbox(X, Y)
    
#    print "Random Walk:" 
#    
#    best_goal, best_estimate, goals, estimates = randwalk(func=eggbox, dims=2, \
#                        lower=(-20,-20), upper=(20,20), \
#                        step_sizes=(10,10), maxit=500, threshold=0.4)
#        
#    print "  Best solution:", best_goal
#    print "  at:", best_estimate
#        
#    fig = pylab.figure()
#    
#    pylab.title("Random Walk Eggbox")
#    pylab.contourf(X, Y, Z, 40)
#    pylab.colorbar()
#    
#    pylab.plot(estimates.T[0], estimates.T[1], '*k')   
#    pylab.plot(0, 0, 'oy', label='global minimum')    
#    pylab.plot(best_estimate[0], best_estimate[1], '^c', label='best solution') 
#    pylab.legend(numpoints=1, prop={'size':9})
#        
#    pylab.xlabel("X")
#    pylab.ylabel("Y")
#    pylab.xlim(-25, 25)
#    pylab.ylim(-25, 25)
#    
#    pylab.figure()
#    pylab.title("Random Walk Goal function")
#    pylab.plot(goals, '-k')
#    pylab.xlabel("Iteration")
    
    
    print "Simulated Annealing:" 
    
    best_goal, best_estimate, goals, estimates = simulated_annealing( \
        func=eggbox, dims=2, lower=(bot,bot), upper=(top,top), \
        step_sizes=step_s, tstart=1000, tfinal=1, tstep=1, it_per_t=100)
        
    print "  Best solution:", best_goal
    print "  at:", best_estimate
        
    fig = pylab.figure()
    
    pylab.title("Simulated Annealing Eggbox")
    pylab.contourf(X, Y, Z, 40)
    pylab.colorbar()
    
    pylab.plot(estimates.T[0], estimates.T[1], '.k')   
    pylab.plot(0, 0, 'oy', label='global minimum')    
    pylab.plot(best_estimate[0], best_estimate[1], 'sc', label='best solution')
    pylab.plot(estimates.T[0][0], estimates.T[1][0], '^m', label='starting solution') 
    pylab.legend(numpoints=1, prop={'size':9})
        
    pylab.xlabel("X")
    pylab.ylabel("Y")
    pylab.xlim(bot, top)
    pylab.ylim(bot, top)
    
    pylab.figure()
    pylab.title("Simulated Annealing Goal function")
    pylab.plot(goals, '-k')
    pylab.xlabel("Iteration")
    
    pylab.show()
    