from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
import matplotlib.pyplot as plt
from Halton_korrekt import Halton_2D
import math


def f(x, y):
    return 1+(1/4000)*x**2+(1/4000)*y**2-cos(x)*cos((1/2)*y*2**(1/2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()