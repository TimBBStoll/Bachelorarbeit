# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Halton_korrekt import halton
from Uniform_Grid_10d import s
from Voronoi.CVT_test_1 import gx
from Voronoi.CVT_test_1 import gy

import Halton_korrekt
import random_korrekt

UG_0=s[:,0]
UG_1=s[:,1]

rand=np.random.rand(100,2)
rd_0 = rand[:,0]
rd_1 = rand[:,1]

halt=halton(10,100)
HA_0= halt[:0]
HA_1= halt[:1]
HA_7= halt[:6]
HA_9= halt[:8]





