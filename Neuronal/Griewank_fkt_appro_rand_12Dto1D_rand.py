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
 return 1+(1/4000)*(a**2+b**2+c**2+d**2+e**2+f**2+g**2+h**2+i**2+j**2+k**2+l**2)-cos(a)*cos((1/2)*b*2**(1/2))*cos((1/3)*c*3**(1/2))*cos((1/4)*d*4**(1/2))*cos((1/5)*e*5**(1/2))*cos((1/6)*f*6**(1/2))*cos((1/7)*g*7**(1/2))*cos((1/8)*h*8**(1/2))*cos((1/9)*i*9**(1/2))*cos((1/10)*j*10**(1/2))*cos((1/11)*k*11**(1/2))*cos((1/12)*l*12**(1/2))
test_data=x
test_targets=objective1(a,b,c,d,e,f,g,h,i,j,k,l)
model= tf.keras.models.load_model("Griewank_fkt_appro_rand_12Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)