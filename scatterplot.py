# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell
import matplotlib.pyplot as plt
import random
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
from matplotlib import cm
from matplotlib import colors
from Halton_korrekt import Halton_2D
from UniformGrid.Uniform_Grid_2D_Probe import UG_2dP

def scatter_plot_2d(x_in: np.ndarray,y_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (-1, 1),
                    lim_z: tuple = (0, 1), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                    title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                    folder_name: str = "figures", show_fig: bool = True, color_map: int = 0):
    '''
    brief: Compute a scatter plot
    input: x_in = [x1,x2] function arguments
           y_in = function values
    return: True if exit successfully
    '''
    # choose colormap
    if color_map == 1:
        c_map = cm.summer
    else:
        c_map = cm.hot

    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    x = x_in
    y = y_in
    z = z_in
    if log:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map, norm=colors.LogNorm())
    else:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map)
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect('auto')
    cbar = fig.colorbar(out, ax=ax, extend='both')
    if show_fig:
        plt.show()
    #plt.savefig(folder_name + "/" + name + ".png", dpi=150)
    return 0
gitter=UG_2dP
x = gitter[:,0]
y = gitter[:,1]

def objective(a, b):
 return a+b
Z=objective(x, y)

#x=x.reshape(10,1)
#y=y.reshape(10,1)
#a=scatter_plot_2d(UG_2dP[:,0],UG_2dP[:,1],Z,lim_x=(0,1),lim_y=(0,1),log=True,color_map=0)

