# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np;  # NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt  # for plotting
from scipy.spatial import Voronoi, voronoi_plot_2d #for voronoi tessellation
import random
import random2
from Halton_korrekt import halton

plt.close('all');  # close all figures

# Simulation window parameters
xMin = 0;
xMax = 1;
yMin = 0;
yMax = 1;
# rectangle dimensions
xDelta = xMax - xMin; #width
yDelta = yMax - yMin #height
areaTotal = xDelta * yDelta; #area of similation window

# Point process parameters
lambda0 = 10;  # intensity (ie mean density) of the Poisson process

# Simulate a Poisson point process
numbPoints = np.random.poisson(lambda0 * areaTotal);  # Poisson number of points

xx = xDelta * np.random.uniform(0, 1, numbPoints) + xMin;  # x coordinates of Poisson points
yy = yDelta * np.random.uniform(0, 1, numbPoints) + yMin;  # y coordinates of Poisson points

xxyy=np.stack((xx,yy), axis=1); #combine x and y coordinates
a=halton(2,100)
##Perform Voroin tesseslation using built-in function
voronoiData=Voronoi(a);
vertexAll=voronoiData.vertices; #retrieve x/y coordinates of all vertices
cellAll=voronoiData.regions; #may contain empty array/set
print(voronoiData)

####START -- Plotting section -- START###
#create voronoi diagram on the point pattern
voronoi_plot_2d(voronoiData, show_points=False,show_vertices=False);
#plot underlying point pattern (ie a realization of a Poisson point process)
#plt.scatter(xx, yy, edgecolor='b', facecolor='b');
#number the points
#for ii in range(numbPoints):
    #plt.text(xx[ii]+xDelta/50, yy[ii]+yDelta/50, ii);
plt.show()


