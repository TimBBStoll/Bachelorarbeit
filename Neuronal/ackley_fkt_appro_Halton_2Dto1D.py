# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from numpy import asarray
import matplotlib.pyplot as plt

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
from matplotlib import cm
from scatterplot import scatter_plot_2d
from matplotlib import colors

x = Halton_2D

x1= np.transpose(x)
#samplefkt f(x)=polyfkt
z=(x1[0,:]**2+x1[1,:]**2)
z=np.reshape(z, (4096, 1))
a=(x1[0,:]**2)
a=np.reshape(a, (4096, 1))
b=(x1[1,:]**2)
b=np.reshape(b, (4096, 1))



y=(-20 * exp(-0.2 * sqrt(0.5 * (z**2)))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b))) + e + 20)


testidx=random.sample(range(4096), 400)
xtest=x1[:,testidx]
ytest=y[testidx]
trainidx=[i for i in range(4096)if i not in testidx]
xtrain=x1[:,trainidx]
ytrain=y[trainidx]

train_data= xtrain
train_targets= ytrain
test_data= xtest
test_targets= ytest

train_data= train_data.reshape(3696,2)
train_targets= train_targets.reshape(3696,1)
test_data= test_data.reshape(400,2)
test_targets= test_targets.reshape(400,1)


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu',))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse', metrics=['mae'])
    return model

model= build_model()
model.fit(train_data,train_targets,epochs=21,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)

y=model.predict(test_data)
a=scatter_plot_2d(train_data[:,0],train_data[:,1],train_targets,lim_x=(0,1),lim_y=(0,1),log=False,color_map=1)
b=scatter_plot_2d(test_data[:,0],test_data[:,1],y,lim_x=(0,1),lim_y=(0,1),log=False,color_map=1)


pointwise_err=np.linalg.norm(y-test_targets,axis=(1))
plt.plot(test_data,pointwise_err,"*")
plt.yscale("log")
plt.show()

