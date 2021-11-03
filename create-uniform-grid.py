"""
Creating a Uniform Grid
~~~~~~~~~~~~~~~~~~~~~~~

Create a simple uniform grid from a 3D NumPy array of values.

"""

import pyvista as pv
import numpy as np


values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
values.shape
print(values)


grid = pv.UniformGrid()


grid.dimensions = np.array(values.shape) + 1


grid.origin = (100, 33, 55.6)
grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!

# Now plot the grid!
grid.plot(show_edges=True)



values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
values.shape
grid = pv.UniformGrid()


grid.dimensions = values.shape
grid.origin = (100, 33, 55.6)
grid.spacing = (1, 5, 2)


grid.point_arrays["values"] = values.flatten(order="F")

grid.plot(show_edges=True)
