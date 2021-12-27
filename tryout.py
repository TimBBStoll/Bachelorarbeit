
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
from UniformGrid.Uniform_Grid_2D_N import UG_2dN
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
import numba
from numba import cuda
from numba import jit
import numpy as np
from timeit import default_timer as timer
x = UG_2dN

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
    model.compile(optimizer='rmsprop',loss='mse', metrics=['mae'])
    return model

model= build_model()
model.fit(train_data,train_targets,epochs=20,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)


gitter=xtrain
gtest=xtest
p1 = gitter[0,:]
p1n=10*p1
listp1=list(p1n)
p2 = gitter[1,:]
p2n=10*p2
listp2=list(p2n)
g1 = gtest[0,:]
listg1=list(g1)
g2 = gtest[1,:]
listg2=list(g2)

def objective1(a, b):
 return 1+(1/4000)*a**2+(1/4000)*b**2-cos(a)*cos((1/2)*b*2**(1/2))

Z=objective1(p1, p2)
listZ=list(Z)


X11, Y11 = meshgrid(p1, p2)
Z11= objective1(X11,Y11)
plotx,ploty = np.meshgrid(np.linspace(np.min(listp1),np.max(listp1),3696),\
                           np.linspace(np.min(listp2),np.max(listp2),3696))
plotz = interp.griddata((listp1,listp2),listZ,(plotx,ploty),method='linear')
plotz1=plotz.reshape(3696,3696)

y1=model.predict(test_data)
list_y1=list(y1)

plotx2,ploty2 = np.meshgrid(np.linspace(np.min(listg1),np.max(listg1),400),\
                           np.linspace(np.min(listg2),np.max(listg2),400))
plotz2 = interp.griddata((listg1,listg2),list_y1,(plotx2,ploty2),method='linear')
plotz22=plotz2.reshape(400,400)


#avc=scatter_plot_2d(p1,p2,Z,lim_x=(0,1),lim_y=(0,1),log=False,color_map=0)
#bvc=scatter_plot_2d(g1,g2,y1,lim_x=(0,1),lim_y=(0,1),log=False,color_map=0)

avc=surface_plot_2d(plotx,ploty,plotz1,lim_x=(0,10),lim_y=(0,10),log=False)
#bvc=surface_plot_2d1(plotx2,ploty2,plotz22,lim_x=(0,1),lim_y=(0,1),log=False)

pointwise_err=np.linalg.norm(y1-test_targets,axis=(1))
plt.plot(test_data,pointwise_err,"*")
plt.yscale("log")
plt.show()
