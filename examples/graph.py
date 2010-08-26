import pylab

from combinatorial import Graph


g = Graph()

g.create_from_points('points.txt', type='full', coords='geo')

for node in g.nodes:
    
    print "Node '%s' connected to:" % (node.key)
    
    for con, w in zip(node.connections, node.weights):
        
        print "  %s - %g" % (con, w)