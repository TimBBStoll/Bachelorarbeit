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
 return (-20 * exp(-0.2 * sqrt(0.5 * (x**2)))-exp(0.5 * (cos(2 * pi * x))) + np.e + 20)
test_data=x
test_targets=objective1(x)
model= tf.keras.models.load_model("ackley_fkt_appro_rand_1Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)