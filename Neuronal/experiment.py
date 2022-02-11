import random
from math import *
from keras import models
from keras import layers
from Halton_korrekt import Halton_2D
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
from Halton_korrekt import Halton_3D
import matplotlib.tri as mtri
from scatterplot import scatter_plot_2d
from surfaceplot import surface_plot_2d
from surfaceplot2 import surface_plot_2d1
import matplotlib.pyplot as plt
import tensorflow as tf
x = Halton_3D

x1= np.transpose(x)
#samplefkt f(x)=sigmoid(x)
z=(x1[0,:]**2+x1[1,:]**2+x1[2,:]**2)
z=np.reshape(z, (4096, 1))
a=(x1[0,:])
a=np.reshape(a, (4096, 1))
b=(x1[1,:])
b=np.reshape(b, (4096, 1))
c=(x1[2,:])
c=np.reshape(c, (4096, 1))
y=(-20 * exp(-0.2 * sqrt(0.5 * (z)))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b)+cos(2 * pi * c))) + e + 20)
testidx=random.sample(range(4096), 400)
xtest=x1[:,testidx]
ytest=y[testidx]

trainidx1=[i for i in range(4096)if i not in testidx]

validx= random.sample(trainidx1, 600)
xval=x1[:,validx]
yval=y[validx]

trainidx=[i for i in range(4096)if i not in testidx+validx]
xtrain=x1[:,trainidx]
ytrain=y[trainidx]

train_data= xtrain
train_targets= ytrain
val_data= xval
val_targets= yval
test_data= xtest
test_targets= ytest

train_data= np.transpose(train_data)
#train_targets= np.transpose(train_targets)
val_data= np.transpose(val_data)
#val_targets= np.transpose(val_targets)
test_data= np.transpose(test_data)
#test_targets= np.transpose(test_targets)

model= tf.keras.models.load_model("ackley_fkt_appro_Halton_2Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)
