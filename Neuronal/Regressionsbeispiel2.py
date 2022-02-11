# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell


import matplotlib.pyplot as plt

import random
from math import *
from keras import models
from keras import layers
from UniformGrid.Uniform_Grid_2D_N import UG_2dN
from UniformGrid.Uniform_Grid_2D_Probe import UG_2dP
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import scipy.interpolate as interp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scatterplot import scatter_plot_2d
from surfaceplot import surface_plot_2d
from surfaceplot2 import surface_plot_2d1
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8
import matplotlib.tri as mtri
from pylab import imshow, show
from timeit import default_timer as timer
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from UniformGrid.Uniform_Grid_2D_N import UG_2dN

x = 10*UG_2dN

x1= np.transpose(x)
#samplefkt f(x)=polyfkt
z=(x1[0,:]**2+x1[1,:]**2)
z=np.reshape(z, (4096, 1))
a=(x1[0,:])
a=np.reshape(a, (4096, 1))
b=(x1[1,:])
b=np.reshape(b, (4096, 1))

y=1+(1/4000)*a**2+(1/4000)*b**2-cos(a)*cos((1/2)*b*2**(1/2))


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

train_data= train_data.reshape(3096,2)
train_targets= train_targets.reshape(3096,1)
val_data= val_data.reshape(600,2)
val_targets= val_targets.reshape(600,1)
test_data= test_data.reshape(400,2)
test_targets= test_targets.reshape(400,1)

score=[]
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu',))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse', metrics=['mae'])
    return model
k=1
num_val_samples = len(val_data)
num_epochs=100
all_mae_histories=[]

model = build_model()
model.fit(train_data,train_targets, epochs=num_epochs, batch_size=16, verbose=0)

history=model.fit(train_data,train_targets,validation_data=(val_data, val_targets), epochs=num_epochs,batch_size=16,verbose=0)
print(history.history.keys())
mae_history = history.history['val_mae']
all_mae_histories.append(mae_history)

average_mae_history=[
   np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history)


plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochen')
plt.ylabel('Mittlerer absoluter Fehler Validierung')
plt.show()

model= build_model()
model.fit(train_data,train_targets,epochs=50,batch_size=1,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mae_score)