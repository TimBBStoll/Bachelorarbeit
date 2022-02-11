# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell


import random
from math import *

import matplotlib.pyplot
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
from UniformGrid.Uniform_Grid_2D_N import UG_2dN
import matplotlib.tri as mtri
from scatterplot import scatter_plot_2d
from surfaceplot import surface_plot_2d
from surfaceplot2 import surface_plot_2d1
import matplotlib.pyplot as plt
import tensorflow as tf
from random_korrekt import rand_2dp
from random_korrekt import rand_2dp1
x = rand_2dp

x1= np.transpose(x)
#samplefkt f(x)=polyfkt
z=(x1[0,:]**2+x1[1,:]**2)
z=np.reshape(z, (36, 1))
a=(x1[0,:])
a=np.reshape(a, (36, 1))
b=(x1[1,:])
b=np.reshape(b, (36, 1))



y=(-20 * exp(-0.2 * sqrt(0.5 * (z)))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b))) + e + 20)
testidx=random.sample(range(36), 4)
xtest=x1[:,testidx]
ytest=y[testidx]

trainidx1=[i for i in range(36)if i not in testidx]

validx= random.sample(trainidx1, 6)
xval=x1[:,validx]
yval=y[validx]

trainidx=[i for i in range(36)if i not in testidx+validx]
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

score=[]
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu',))
    model.add(layers.Dense(64, activation='relu', ))
    model.add(layers.Dense(64, activation='relu', ))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse', metrics=['mae'])
    return model
k=1
num_val_samples = len(val_data)
num_epochs=10
all_mae_histories=[]

model = build_model()
model.summary()
mc_best = tf.keras.callbacks.ModelCheckpoint("ackley_fkt_appro_UG_2Dto1D_layer91" + '/best_model', monitor='val_loss', mode='min',
                                                     save_best_only=True, verbose=0)
#history=model.fit(train_data,train_targets,validation_data=(val_data, val_targets), epochs=1000,batch_size=100,verbose=2,callbacks=mc_best)
#history=model.fit(train_data,train_targets,validation_data=(val_data, val_targets), epochs=1000,batch_size=300,verbose=2,callbacks=mc_best)
history=model.fit(train_data,train_targets,validation_data=(val_data, val_targets), epochs=num_epochs,batch_size=256,verbose=0,callbacks=mc_best)

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
smooth_mae_history = smooth_curve(average_mae_history[3:])


plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochen')
plt.ylabel('Mittlerer absoluter Fehler Validierung')
#plt.show()

model= tf.keras.models.load_model("ackley_fkt_appro_UG_2Dto1D_layer91/best_model")
probe1=rand_2dp1
x90= np.transpose(probe1)
a23=(x90[0,:])
a24=(x90[1,:])
def objective1(a,b):
 return (-20 * exp(-0.2 * sqrt(0.5 * ((a**2+b**2))))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b))) + np.e + 20)
bnn=objective1(a23,a24)
fgg= np.transpose(x90)
test_mse_score,test_mae_score=model.evaluate(fgg,bnn)
print(test_mse_score)
print(test_mae_score)
