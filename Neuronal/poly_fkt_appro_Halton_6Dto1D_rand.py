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
 return 30*(a)**(4)-60*(a)**(3)+35*(a)**(2)-4.5*(a)+30*(b)**(4)-60*(b)**(3)+35*(b)**(2)-4.5*(b)+30*(c)**(4)-60*(c)**(3)+35*(c)**(2)-4.5*(c)+30*(d)**(4)-60*(d)**(3)+35*(d)**(2)-4.5*(d)+30*(e)**(4)-60*(e)**(3)+35*(e)**(2)-4.5*(e)+30*(f)**(4)-60*(f)**(3)+35*(f)**(2)-4.5*(f)*cos((1/4)*d*4**(1/2))*cos((1/5)*e*5**(1/2))*cos((1/6)*f*6**(1/2))
test_data=x
test_targets=objective1(a,b,c,d,e,f)
model= tf.keras.models.load_model("poly_fkt_appro_Halton_6Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)