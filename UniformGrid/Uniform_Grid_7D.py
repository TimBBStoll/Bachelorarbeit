# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
size = 2
dim =7
s_d = np.linspace ( 0.0, 1.0, size)


sx_0 = np.zeros ( size**dim )
sx_1 = np.zeros ( size**dim )
sx_2 = np.zeros ( size**dim )
sx_3 = np.zeros ( size**dim )
sx_4 = np.zeros ( size**dim )
sx_5 = np.zeros ( size**dim )
sx_6 = np.zeros ( size**dim )



s = np.zeros ( [ size**dim, dim ] )

k = 0
for p in range(0,size):
  for o in range(0, size):
    for n in range(0, size):
            for m in range (0,size):
              for l in range (0,size):
                for j in range ( 0, size ):
                  for i in range ( 0, size ):
                    sx_0[k] = s_d[i]
                    sx_1[k] = s_d[j]
                    sx_2[k] = s_d[l]
                    sx_3[k] = s_d[m]
                    sx_4[k] = s_d[n]
                    sx_5[k] = s_d[o]
                    sx_6[k] = s_d[p]


                    s[k,0] = s_d[i]
                    s[k,1] = s_d[j]
                    s[k,2] = s_d[l]
                    s[k,3] = s_d[m]
                    s[k,4] = s_d[n]
                    s[k,5] = s_d[o]
                    s[k,6] = s_d[p]

                    k = k + 1
UG_7d=s
x0=s[:,0]
x1=s[:,1]
x2=s[:,2]
x3=s[:,3]
x4=s[:,4]
x5=s[:,5]
x6=s[:,6]

#plt.scatter(x0, x1)

#plt.show()