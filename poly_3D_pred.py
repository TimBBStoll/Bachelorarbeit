# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell

from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from scatterplot import scatter_plot_2d
from surfaceplot import surface_plot_2d
from surfaceplot2 import surface_plot_2d1
import matplotlib.pyplot as plt

def objective(a, b):
 return 30*(a)**(4)-60*(a)**(3)+35*(a)**(2)-4.5*(a)+30*(b)**(4)-60*(b)**(3)+35*(b)**(2)-4.5*(b)


r_min, r_max = 0, 1
xaxis = arange(r_min, r_max, 0.002)
yaxis = arange(r_min, r_max, 0.002)
x, y = meshgrid(xaxis, yaxis)
results = objective(x, y)
figure = plt.figure(figsize=(5.8, 4.7), dpi=400)
axis = figure.gca( projection='3d')
axis.plot_surface(x, y, results, cmap='jet', shade= "false")
plt.show()
#plt.contour(x,y,results)
#plt.show()
#plt.scatter(x, y, results)
#plt.show()
bvc=scatter_plot_2d(x,y,results,lim_x=(0,1),lim_y=(0,1),log=False,color_map=0,label_x= "Halton_2D", label_y="",title= "Poly 2D")