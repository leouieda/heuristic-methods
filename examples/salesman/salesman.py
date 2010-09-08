"""
Solve a sample Traveling Salesman problem with the Brazilian capitals
"""

import time

import numpy
import pylab
from mpl_toolkits.basemap import Basemap

import hm.salesman


# LOAD THE CITY DATA FROM THE FILE
filename = 'brasil_capitals.txt'
print "Loading city data from file %s" % (filename)
cityfile = open(filename, 'r')

cities = []
citynames = []

#filestring = cityfile.read()

for line in cityfile:
    
    if len(line) == 0:
        
        continue
    
    cityname, lon_string, lat_string = line.split(',')
    
    citynames.append(cityname)
    
    deg, min, sec = lon_string.split(':')
    
    lon = float(deg)
    
    if lon >= 0:
        
        lon += float(min)/60. + float(sec)/3600.
        
    else:
                
        lon -= float(min)/60. + float(sec)/3600.        
    
    deg, min, sec = lat_string.split(':')
            
    lat = float(deg)
    
    if lat >= 0:
        
        lat += float(min)/60. + float(sec)/3600.
        
    else:
                
        lat -= float(min)/60. + float(sec)/3600.        

    cities.append((lon, lat))   

ncities = len(cities)

print "  %d city coordinates loaded" % (ncities)

# SET THE STARTING CITY BY PUTTING IT FIRST IN THE LIST
start = 21

tmp = cities[0]
cities[0] = cities[21]
cities[21] = tmp

tmpname = citynames[0]
citynames[0] = citynames[21]
citynames[21] = tmpname

print "Starting point: %s" % (citynames[0])

print "Building the distances table"

dist_table = hm.salesman.build_distance_table(cities, type='geographic')

pylab.savetxt("DMAT.txt", dist_table)

print "Solving..."

tstart = time.time()

result = hm.salesman.solve_ga(dist_table, ncities, pop_size=100, \
                              mutation_prob=0.5, crossover_prob=0.7, \
                              max_it=1000)

tend = time.time()

print "Done in %g seconds" % (tend - tstart)

best_routes, best_dists, dists = result

print "Best total distance:", best_dists[-1]
print "Best route:"

print "  0: %s" % (citynames[0])

for i, city in enumerate(best_routes[-1]):
    
    print "  %d: %s" % (i + 1, citynames[city])

# PLOTS OF THE DISTANCE PER GENERATION
pylab.figure(figsize=(13,6))
pylab.suptitle("Traveling Salesman: Genetic Algorithm")

pylab.subplot(2,2,1)
pylab.plot(dists, '.-k', label="Generation best")
pylab.ylabel("Distance")
pylab.legend(prop={'size':10})

pylab.subplot(2,2,3)
pylab.plot(best_dists, '.-k', label="Best ever")
pylab.xlabel("Iteration")
pylab.ylabel("Distance")
pylab.legend(prop={'size':10})

# PLOT THE BEST ROUTE FOUND
ax = pylab.subplot(1,2,2)
pylab.axis('scaled')
pylab.title("Best Route: Distance = %g km" % (best_dists[-1]))

xmin, xmax, ymin, ymax = -70.0, -20.0, -40.0, 10.0

bm = Basemap(projection='merc', \
             llcrnrlon=xmin, llcrnrlat=ymin, \
             urcrnrlon=xmax, urcrnrlat=ymax, \
             lon_0=0.5*(xmax + xmin), lat_0=0.5*(ymax + ymin), \
             resolution='l', area_thresh=1000000)

dlon = 10.
dlat = 10.
bm.drawmeridians(numpy.arange(xmin, xmax, dlon), \
                 labels=[0,0,0,1], fontsize=12, linewidth=1, rotation=45)
bm.drawparallels(numpy.arange(ymin + dlat, ymax + dlat, dlat), \
                 labels=[1,0,0,0], fontsize=12, linewidth=1)
    
bm.drawcoastlines(linewidth=1.5)
#bm.fillcontinents(color='green')
#bm.drawmapboundary()
bm.bluemarble()
bm.drawcountries(linewidth=2)
bm.drawstates(linewidth=1)

city = 0
for next in best_routes[-1]:
    
    path_x = [cities[city][0], cities[next][0]]
    path_y = [cities[city][1], cities[next][1]]
    
    path_x, path_y = bm(path_x, path_y)
    
    bm.plot(path_x, path_y, 'o-w')
    
    x, y = bm(cities[city][0], cities[city][1])
    
    ax.text(x, y, citynames[city], fontsize=8, color='w')

    city = next
    
x, y = bm(cities[city][0], cities[city][1])

ax.text(x, y, citynames[city], fontsize=8, color='w')

path_x = [cities[city][0], cities[0][0]]
path_y = [cities[city][1], cities[0][1]]
path_x, path_y = bm(path_x, path_y)

bm.plot(path_x, path_y, 'o-w', label="Cities")

x, y = bm(cities[0][0], cities[0][1])
bm.plot(x, y, 'oy', label='Starting point')

pylab.legend(loc='lower left', prop={'size':9}, numpoints=1)

pylab.show()