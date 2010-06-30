"""
Collection of Monte Carlo optimizers 
"""

import math

import numpy
import pylab


def normal_rand(mean, stddev):
    """
    Return sample from a Normal distribution.
    """
    
    u1 = numpy.random.uniform(-1, 1)
    
    u2 = numpy.random.uniform(-1, 1)
    
    r = u1**2 + u2**2
    
    while r >= 1.0:

        u1 = numpy.random.uniform(-1, 1)
    
        u2 = numpy.random.uniform(-1, 1)
    
        r = u1**2 + u2**2    

    return mean + stddev*u2*math.sqrt(-2*math.log(r)/r)
    

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


def mutate(value, lower, upper, size):
    """
    Mutate value in the domain [lower,upper] with mutation 'size' as a decimal 
    percentage of the domain.
    WARNING:
        Doesn't work if the mutation size is too big and breaches both ends
        of the domain! 
    """
    
    # Define a random size for the mutation in the domain
    deltav = size*(upper - lower)
    
    vmax = value + deltav
    
    vmin = value - deltav

    if vmax > upper:
        
        # Percentage of the the total mutation size that fell in the domain
        bias = (upper - value)/deltav
        
        mutated = numpy.random.uniform(vmin, upper)
        
        # This compensated for the smaller region after 'value'
        while not (mutated >= value or flip(bias)):
        
            mutated = numpy.random.uniform(vmin, upper)
            
    elif vmin < lower:
        
        # Do the same as above but in the case where the mutation size breaches
        # the lower bound of the domain
        bias = (value - lower)/deltav
                        
        mutated = numpy.random.uniform(lower, vmax)
        
        # This compensated for the smaller region after 'value'
        while not (mutated <= value or flip(bias)):
        
            mutated = numpy.random.uniform(lower, vmax)
            
    else:
        
        # If the mutation size doesn't breach the domain bounds, just make a 
        # random mutation        
        mutated = numpy.random.uniform(vmin, vmax)
        
    return mutated


def mutated_rw(func, lower, upper, num_agentes, prob_mutation=0.01, \
               size_mutation=0.1, max_it=100, threshold=0.8):
    """
    Random Walk with mutations.
    """
    
    # Generate random agents
    agents = numpy.zeros(num_agentes)
    
    goals = numpy.zeros(num_agentes)
    
    for i in xrange(num_agentes):
        
        agents[i] = numpy.random.uniform(lower, upper)
        
        goals[i] = func(agents[i])
        
        if i == 0 or best_goal > goals[i]:
            
            best_agent = agents[i]
            
            best_goal = goals[i]
            
    survivors = []
    
    for iteration in xrange(max_it):
            
        survivors = []
        
        # Kill all the unworthy! (according to a survival probability)        
        for i in xrange(num_agentes):
                        
            survival_prob = math.exp(-abs((goals[i] - min(goals))/min(goals)))
            
            if flip(survival_prob):
                
                survivors.append(agents[i])
                
        if len(survivors) >= threshold*num_agentes:
            
            print "Enough survivors: %d out of %d" % (len(survivors), \
                                                      num_agentes)
            
            break
                
        # Mutate the survivors
        for i in xrange(len(survivors)):
            
            if flip(prob_mutation):
            
                agents[i] = mutate(survivors[i], lower, upper, size_mutation)
            
            else:
                
                agents[i] = survivors[i]
            
            goals[i] = func(agents[i])
            
            if best_goal > goals[i]:
                
                best_agent = agents[i]
                
                best_goal = goals[i]
            
        for i in xrange(len(survivors), num_agentes):
            
            agents[i] = numpy.random.uniform(lower, upper)
            
            goals[i] = func(agents[i])
            
            if best_goal > goals[i]:
                
                best_agent = agents[i]
                
                best_goal = goals[i]        
    
    print "Needed %d iterations." % (iteration + 1)
    
    return best_agent, best_goal, agents, goals
            
                
                 
    
            

def randwalk(func, dims, lower, upper, num_solutions, threshold=0.40):
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
        
    # Keep all the steps recorded
    estimates = []
    
    goals = []
    
    solution_i = 0
        
    while solution_i < threshold*num_solutions:
        
        for j in xrange(num_solutions):
            
            tmp = numpy.zeros(dims)
            
            for i in xrange(dims):
                
                tmp[i] = numpy.random.uniform(lower[i], upper[i])
                
            goal = func(*tmp)
        
            if solution_i == 0:
                
                survival_prob = 1
                
            else:
                
                survival_prob = abs((max(goals) - goal)/(max(goals) - min(goals)))
            
            if flip(survival_prob):
                
                estimate = tmp
                
                if solution_i == 0 or goal < best_goal:
                    
                    best_goal = goal
                    
                    best_estimate = estimate
                
                estimates.append(estimate)
                
                goals.append(goal)
                
                solution_i += 1
                
                if solution_i == num_solutions:
                    
                    break
                
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
    
    time_steps = numpy.arange(0, tfinal+1, 1, 'f')
    
    cooling_schedule = tstart*0.99**(time_steps)

    for temperature in cooling_schedule:
       
        accepted = 0
        
        for iteration in xrange(it_per_t):
            
            # Make a random perturbation
            for i in xrange(dims):
                
                step[i] = numpy.random.uniform(-step_sizes[i], step_sizes[i])
                
                # If the step takes the estimate out of the bounds, bring it
                # back.
                if estimate[i] + step[i] > upper[i]:
                    
                    step[i] = -step[i]
                    
                if estimate[i] + step[i] < lower[i]:
                                        
                    step[i] = -step[i]           
                    
            tmp = estimate + step
            
            goal = func(*tmp)
            
            delta_goal = goals[-1] - goal
            
            survival_prob = math.exp(delta_goal/temperature)
            
            if delta_goal > 0 or flip(survival_prob):
                                            
                estimate = tmp
                
                if goal < best_goal:
                    
                    best_goal = goal
                    
                    best_estimate = estimate        
                
                estimates.append(estimate)
                
                goals.append(goal)
                
                accepted += 1
                
        if accepted == 0:
            
            print "Exited due to frozen system: temp = %g" % (temperature)
            
            break
        
    return best_goal, best_estimate, goals, numpy.array(estimates)     
    
    
    
if __name__ == '__main__':
        
    import pylab
    from test_functions import eggbox
    import gradient
    
    bot = -20
    top = 20
    step = (top - bot)/(100. - 1.)
        
    x = numpy.arange(bot, top + step, step)
    y = numpy.arange(bot, top + step, step)    
    X, Y = pylab.meshgrid(x, y)
    Z = eggbox(X, Y)
        
#    print "Random Walk:" 
#    
#    best_goal, best_estimate, goals, estimates = randwalk(func=eggbox, dims=2, \
#                        lower=(-20,-20), upper=(20,20), \
#                        num_solutions=500, threshold=0.8)
#        
#    print "  Best solution of", len(goals), ":", best_goal
#    print "  at:", best_estimate
#        
##    print "  All solutions:\n", numpy.array([estimates.T[0], estimates.T[1], goals]).T
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
#    pylab.xlim(bot, top)
#    pylab.ylim(bot, top)
#    
#    pylab.figure()
#    pylab.title("Random Walk Goal function")
#    pylab.plot(goals, '-k')
#    pylab.xlabel("Iteration")
    
    
    print "Simulated Annealing:" 
    
    best_goal, best_estimate, goals, estimates = simulated_annealing( \
        func=eggbox, dims=2, lower=(bot,bot), upper=(top,top), \
        step_sizes=(0.1,0.1), tstart=1500, tfinal=1500, tstep=1, it_per_t=20)
        
    print "  Best solution of %d: %g" % (len(goals), best_goal)
    print "  at:", best_estimate
        
    fig = pylab.figure()
    
    pylab.title("Simulated Annealing Eggbox")
    pylab.pcolor(X, Y, Z)
    pylab.colorbar()
    
    pylab.plot(estimates.T[0], estimates.T[1], '-k')   
    pylab.plot(0, 0, 'oy', label='global minimum')    
    pylab.plot(best_estimate[0], best_estimate[1], 'sc', label='best solution')
    pylab.plot(estimates.T[0][-1], estimates.T[1][-1], 'sm', label='final solution')
    pylab.plot(estimates.T[0][0], estimates.T[1][0], '^m', label='starting solution') 
    pylab.legend(numpoints=1, prop={'size':9})
        
    pylab.xlabel("X")
    pylab.ylabel("Y")
    pylab.xlim(bot, top)
    pylab.ylim(bot, top)
    
    pylab.figure()
    pylab.title("Simulated Annealing Goal Function")
    pylab.plot(goals, '-k')
    pylab.xlabel("Iteration")
    
#    print "Improve solution with Gauss-Newton:"
#    
#    best_estimate, best_goal, estimates, goals = gradient.newton(\
#        func=eggbox, dims=2, initial=best_estimate, maxit=100, stop=10**(-15))
#        
#    print "  Best solution: %g" % (best_goal)
#    print "  at:", best_estimate
#        
#    fig = pylab.figure()
#    
#    pylab.title("Gauss-Newton Eggbox")
#    pylab.contourf(X, Y, Z, 40)
#    pylab.colorbar()
#    
#    pylab.plot(estimates.T[0], estimates.T[1], '-k')   
#    pylab.plot(0, 0, 'oy', label='global minimum')    
#    pylab.plot(best_estimate[0], best_estimate[1], 'sc', label='best solution') 
#    pylab.plot(estimates.T[0][0], estimates.T[1][0], '^m', label='starting solution') 
#    pylab.legend(numpoints=1, prop={'size':9})
#        
#    pylab.xlabel("X")
#    pylab.ylabel("Y")
#    pylab.xlim(bot, top)
#    pylab.ylim(bot, top)
#    
#    pylab.figure()
#    pylab.title("Gauss-Newton Eggbox Goal Function")
#    pylab.plot(goals, '-k')
#    pylab.xlabel("Iteration")
    
    pylab.show()
    