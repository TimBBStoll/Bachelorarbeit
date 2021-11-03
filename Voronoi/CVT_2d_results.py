# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import r8vec_uniform_01
import numpy as np
import matplotlib.pyplot as plt
import platform
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d
from r8vec_uniform_01 import r8vec_uniform_01
from cvt_2d_sampling import cvt_2d_sampling
x=cvt_2d_sampling(16,20,10)
print(x)
