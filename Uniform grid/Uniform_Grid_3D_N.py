# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
size = 2**4
dim =3
s_d = np.linspace ( 0.0, 1.0, size)


sx_0 = np.zeros ( size**dim )
sx_1 = np.zeros ( size**dim )
sx_2 = np.zeros ( size**dim )





s = np.zeros ( [ size**dim, dim ] )

k = 0
for l in range (0,size):
                for j in range ( 0, size ):
                  for i in range ( 0, size ):
                    sx_0[k] = s_d[i]
                    sx_1[k] = s_d[j]
                    sx_2[k] = s_d[l]



                    s[k,0] = s_d[i]
                    s[k,1] = s_d[j]
                    s[k,2] = s_d[l]



                    k = k + 1
UG_3dN=s
x0=s[:,0]
x1=s[:,1]
x2=s[:,2]


#plt.scatter(x0, x1)

#plt.show()