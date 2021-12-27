# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
size = 6
dim =2
s_d = np.linspace ( 0.0, 1.0, size)


sx_0 = np.zeros ( size**dim )
sx_1 = np.zeros ( size**dim )






s = np.zeros ( [ size**dim, dim ] )

k = 0
for j in range ( 0, size ):
                  for i in range ( 0, size ):
                    sx_0[k] = s_d[i]
                    sx_1[k] = s_d[j]




                    s[k,0] = s_d[i]
                    s[k,1] = s_d[j]




                    k = k + 1
UG_2dP=s
x0=s[:,0]
x1=s[:,1]



#plt.scatter(x0, x1)

#plt.show()