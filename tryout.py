import numpy as np
from math import *

from UniformGrid.Uniform_Grid_2D_N import UG_2dN
x = UG_2dN
x=x.reshape(2,4096)
#print(x.shape)
z=sum(x**2)
test1=np.random.choice(z,400)
#print(test1.shape)


a=np.array([[0.1,0.2,0.3,0.4,0.5],
            [0.1,0.2,0.3,0.4,0.5]])
a1=a**2
print(a1.shape)
a2=a1[:1]
a3=a1[1:2]
a4=a2+a3
a4=a4.reshape(5,)
test=np.random.choice(a4,2,False)
print(test)
p=list(set(a4)-set(test))
print(p)
xp=np.array(p)
print(xp.shape)

