from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
import matplotlib.pyplot as plt
from Halton_korrekt import Halton_1D

x = Halton_1D
x1=np.transpose(x)

y=(-20 * exp(-0.2 * sqrt(0.5 * (x)))-exp(0.5 * (cos(2 * np.pi * x))) + np.e + 20)
print(y)

r_min, r_max = -1, 1
xaxis = arange(r_min, r_max, 2.0)
yaxis = arange(r_min, r_max, 2.0)
x, y = meshgrid(x1[0,:],x1[1,:])
results = f(x, y)
figure = plt.figure()
axis = figure.gca( projection='3d')
axis.plot_surface(x, y, results, cmap='jet', shade= "false")
plt.show()
plt.contour(x,y,results)
plt.show()
plt.scatter(x, y, results)
plt.show()