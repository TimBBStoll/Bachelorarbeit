# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell
import matplotlib.pyplot as plt

from numpy import cos

from numpy import meshgrid
import numpy as np

from UniformGrid.Uniform_Grid_2D_N import UG_2dN

def surface_plot_2d1(x_in: np.ndarray,y_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (0, 10), lim_y: tuple = (0, 10),
                    lim_z: tuple = (0, 10), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
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
        c_map = 'jet'
    else:
        c_map = 'jet'

    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111,projection='3d')
    x = x_in
    y = y_in
    z = z_in

    if log:
        out = ax.plot_surface(x, y, z, cmap='jet', shade= "false", rstride=50, cstride=50)
    else:
        out = ax.plot_surface(x, y, z, cmap='jet', shade= "false", rstride=1, cstride=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    cbar = fig.colorbar(out, ax=ax, extend='both',pad=0.1)
    if show_fig:
        plt.show()
    #plt.savefig(folder_name + "/" + name + ".png", dpi=150)
    return 0
gitter=UG_2dN
x = gitter[:,0]
xn=10*x
y = gitter[:,1]
yn=10*y

X11,Y11 = np.meshgrid(xn,yn)

def objective(a, b):
 return  1+(1/4000)*a**2+(1/4000)*b**2-cos(a)*cos((1/2)*b*2**(1/2))

x1, y1 = meshgrid(x, y)
Z= objective(X11,Y11)

#x=x.reshape(10,1)
#y=y.reshape(10,1)

#a=surface_plot_2d1(X11,Y11,Z,log=False,color_map=1,lim_x=(0,10),lim_y=(0,10))

