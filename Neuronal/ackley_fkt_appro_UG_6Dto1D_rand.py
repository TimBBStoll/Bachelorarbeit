from math import *


from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from random_korrekt import rand_6du
import numpy as np
import tensorflow as tf

x = rand_6du
x1= np.transpose(x)

z=(x1[0,:]**2+x1[1,:]**2+x1[2,:]**2+x1[3,:]**2+x1[4,:]**2+x1[5,:]**2)
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
def objective1(a,b,c,d,e,f):
 return (-20 * exp(-0.2 * sqrt(0.5 * (z)))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b)+cos(2 * pi * c)+cos(2 * pi * d)+cos(2 * pi * e)+cos(2 * pi * f))) + np.e + 20)
test_data=x
test_targets=objective1(a,b,c,d,e,f)
model= tf.keras.models.load_model("ackley_fkt_appro_UG_6Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)