from matplotlib import cm  # color map
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos

X = np.linspace(0, 10, 200)
Y = np.linspace(0, 10, 200)
X, Y = np.meshgrid(X, Y)

Z = 1+(1/4000)*X**2+(1/4000)*Y**2-cos(X)*cos((1/2)*Y*2**(1/2))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, \
                       cstride=1, cmap=cm.jet)

# plt.savefig('rastrigin.png')
plt.show()