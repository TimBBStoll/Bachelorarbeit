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
x=Halton_2D
x1=np.transpose(x)
def griewank_function(x, dim):
    """Griewank's function multimodal, symmetric, inseparable """
    partA = 0
    partB = 1
    for i in range(dim):
        partA += x[i]**2
        partB *= math.cos(float(x[i]) / math.sqrt(i+1))
    return 1 + (float(partA)/4000.0) - float(partB)

x=griewank_function(x,2)
print(x)