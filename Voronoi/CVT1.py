# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import random
import random2 as rd

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d

points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])
vor = Voronoi(points)

fig = voronoi_plot_2d(vor)
plt.show()