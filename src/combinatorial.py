"""
Solvers for the shortest path in a graph.
Depends on graph.Graph class.
"""

import numpy

def random_walk(graph, start, end, num_solutions, threshold=0.8, max_it=100):
    """
    Look for the shortest path in a graph from start to end using a random walk
    
    Parameters:
    
        graph: instance of graph.Graph
        
        start: key identifying the staring node
        
        end: key identifying the end node
        
        num_solutions: how many solutions to generate
        
        threshold: percentage of solutions that
    """