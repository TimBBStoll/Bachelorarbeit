from math import *


from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from random_korrekt import rand_2du
import numpy as np
import tensorflow as tf

x = rand_2du
x1= np.transpose(x)
#samplefkt f(x)=polyfkt
z=(x1[0,:]**2+x1[1,:]**2)
z=np.reshape(z, (400, 1))
a=(x1[0,:])
a=np.reshape(a, (400, 1))
b=(x1[1,:])
b=np.reshape(b, (400, 1))
def objective1(a,b):
 return 30*(a)**(4)-60*(a)**(3)+35*(a)**(2)-4.5*(a)+30*(b)**(4)-60*(b)**(3)+35*(b)**(2)-4.5*(b)
test_data=x
test_targets=objective1(a,b)
model= tf.keras.models.load_model("poly_fkt_appro_Halton_2Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)