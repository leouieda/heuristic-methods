"""
Combinatorial optimization solvers
(for tsp-like problems) 
"""

import numpy
from math import cos, sin, sqrt


class GraphNode():
    """
    A node in a graph. Holds information on it's connections and weights
    """
    
    def __init__(self, key, connections, weights, x=None, y=None, z=None):
        """
        Parameters:
        
            key: key identifying this node in the graph
            
            connections: list of keys of the nodes with which this on has
                         connections
                         
            weights: list with the weights of each connection
            
            x, y, z: position of the node (keep None if does not apply)
        """
        
        assert len(connections) == len(weights), \
            "Must have a weights for each connection of node '%s'!" % (key)

        self.key = key
        
        self.x = x

        self.y = y
        
        self.z = z
        
        self.connections = connections
        
        self.weights = weights
        
        

class Graph():
    """
    A graph class. Make a graph by connecting a set of points.
    """
    
    earth_radius = 6371.
    
    deg2rad = numpy.pi/180.
    
    
    def __ini__(self):
        
        
        self.nodes = []
        
        
    def create_from_points(self, fname, type='full', coords='geo'):
        """
        Create a graph from a list of points in a file.
        
        File structure:
        
            key1, x1, y1        
            key2, x2, y2
            ...
            keyN, xN, yN
        
        key is a string or number that identifies the point. x and y are the
        coordinates of the point.
            
        Parameters:
            
            fname: name of the file with the points
            
            type: kind of graph to make (only full graph supported)
                * full = each node connects to all others
                
            coords: type of coordinates used
                * 'geo' = geographic coordinates (latitude and longitude)
                    x = lon and y = lat           
        """
        
        assert coords in ['geo'], 'Invalid coordinate system'
        
        file = open(fname)
        
        file_string = file.read()
        
        self.nodes =[]
        
        for line in file_string.split('\n'):
            
            key, x_string, y_string = line.split(',')
            
            deg, min, sec = x_string.split(':')
            
            x = self.deg2rad*(float(deg) + float(min)/60. + float(sec)/3600.)
            
            deg, min, sec = y_string.split(':')
            
            y = self.deg2rad*(float(deg) + float(min)/60. + float(sec)/3600.)
            
            connections = []
            
            weights = []
            
            # Connect the new node to all the others
            for node in self.nodes:
                
                connections.append(node.key)
                
                node.connections.append(key)
                
                # Calculate the distance between the nodes
                if coords == 'geo':
                    
                    cos_arc = sin(y)*sin(node.y) + \
                              cos(y)*cos(node.y)*cos(x - node.x)
                    
                    distance = sqrt(2*(self.earth_radius**2)*(1 - cos_arc))
                
                weights.append(distance)
                
                node.weights.append(distance)            
            
            self.nodes.append(GraphNode(key, connections, weights, x, y))
            
    