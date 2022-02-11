from math import *


from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from random_korrekt import rand_12du
import numpy as np
import tensorflow as tf

x = rand_12du
x1= np.transpose(x)

z=(x1[0,:]**2+x1[1,:]**2+x1[2,:]**2+x1[3,:]**2+x1[4,:]**2+x1[5,:]**2+x1[6,:]**2+x1[7,:]**2+x1[8,:]**2+x1[9,:]**2+x1[10,:]**2+x1[11,:]**2)
z=np.reshape(z, (400, 1))

a=(x1[0,:])
a=np.reshape(a, (400, 1))

b=(x1[1,:])
b=np.reshape(b, (400, 1))

c=(x1[2,:])
c=np.reshape(c, (400, 1))

d=(x1[3,:])
d=np.reshape(d, (400, 1))

e=(x1[4,:])
e=np.reshape(e, (400, 1))

f=(x1[5,:])
f=np.reshape(f, (400, 1))

g=(x1[6,:])
g=np.reshape(g, (400, 1))

h=(x1[7,:])
h=np.reshape(h, (400, 1))

i=(x1[8,:])
i=np.reshape(i, (400, 1))

j=(x1[9,:])
j=np.reshape(j, (400, 1))

k=(x1[10,:])
k=np.reshape(k, (400, 1))

l=(x1[11,:])
l=np.reshape(l, (400, 1))


def objective1(a,b,c,d,e,f,g,h,i,j,k,l):
 return 30*(a)**(4)-60*(a)**(3)+35*(a)**(2)-4.5*(a)+30*(b)**(4)-60*(b)**(3)+35*(b)**(2)-4.5*(b)+30*(c)**(4)-60*(c)**(3)+35*(c)**(2)-4.5*(c)+30*(d)**(4)-60*(d)**(3)+35*(d)**(2)-4.5*(d)+30*(e)**(4)-60*(e)**(3)+35*(e)**(2)-4.5*(e)+30*(f)**(4)-60*(f)**(3)+35*(f)**(2)-4.5*(f)+30*(g)**(4)-60*(g)**(3)+35*(g)**(2)-4.5*(g)+30*(h)**(4)-60*(h)**(3)+35*(h)**(2)-4.5*(h)+30*(i)**(4)-60*(i)**(3)+35*(i)**(2)-4.5*(i)+30*(j)**(4)-60*(j)**(3)+35*(j)**(2)-4.5*(j)+30*(k)**(4)-60*(k)**(3)+35*(k)**(2)-4.5*(k)+30*(l)**(4)-60*(l)**(3)+35*(l)**(2)-4.5*(l)
test_data=x
test_targets=objective1(a,b,c,d,e,f,g,h,i,j,k,l)
model= tf.keras.models.load_model("Poly_fkt_appro_Halton_12Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)