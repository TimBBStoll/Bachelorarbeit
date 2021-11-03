# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Halton_korrekt import halton

from tryout_Voronoi import data
from sklearn.metrics import mean_squared_error
import Halton_korrekt
import random_korrekt


size = 10
dim =2
s_d = np.linspace ( 0.0, 1.0, size)


sx_0 = np.zeros ( size**dim )
sx_1 = np.zeros ( size**dim )
S = np.zeros ( [ size**dim, dim ] )

k = 0
for j in range(0, size):
    for i in range(0, size):
        sx_0[k] = s_d[i]
        sx_1[k] = s_d[j]
        S[k, 0] = s_d[i]
        S[k, 1] = s_d[j]
        k = k + 1
UG=S

rand=np.random.rand(100,2)


halt=halton(10,100)
HA_0= halt[:,0:1]
HA_1= halt[:,1:2]
HA_7= halt[:,6:7]
HA_9= halt[:,7:8]

halt2 = halt[:,6:9:2]

CVT=data




print(mean_squared_error(UG, halt2))


