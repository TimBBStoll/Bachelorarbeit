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
import matplotlib.pyplot as plt

def objective(x, y):
 return 1+(1/4000)*x**2+(1/4000)*y**2-cos(x)*cos((1/2)*y*2**(1/2))


r_min, r_max = 0, 1
xaxis = arange(r_min, r_max, 0.002)
yaxis = arange(r_min, r_max, 0.002)
x, y = meshgrid(xaxis, yaxis)
results = objective(x, y)
figure = plt.figure(figsize=(5.8, 4.7), dpi=400)
axis = figure.gca( projection='3d')
axis.plot_surface(x, y, results, cmap='jet', shade= "false")
plt.show()
plt.contour(x,y,results)
plt.show()
plt.scatter(x, y, results)
plt.show()
