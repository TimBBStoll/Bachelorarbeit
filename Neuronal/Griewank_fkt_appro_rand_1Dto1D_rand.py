from math import *


from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from random_korrekt import rand_1du
import numpy as np
import tensorflow as tf

x = rand_1du

#samplefkt f(x)=polyfkt



def objective1(x):
 return 1+(1/4000)*x**2-cos(x)
test_data=x
test_targets=objective1(x)
model= tf.keras.models.load_model("Griewank_fkt_appro_rand_1Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)