
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
model.fit(train_data,train_targets,epochs=150,batch_size=1,verbose=1)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)


gitter=xtrain
gtest=xtest
gtestnew=gtest.transpose()
#gtest_sort = np.sort(gtest,axis=1)
#sortedArr = gtestnew[gtestnew[:,0].argsort()]
#ggg=sortedArr.transpose()
ggg=xtest

p1 = gitter[0,:]
p1n=p1
listp1=list(p1n)

p2 = gitter[1,:]
p2n=p2
listp2=list(p2n)

g1 = ggg[0,:]
g1n=g1
listg1=list(g1n)


g2 = ggg[1,:]
g2n=g2
listg2=list(g2n)


def objective1(a,b):
 return 1+(1/4000)*a**2+(1/4000)*b**2-cos(a)*cos((1/2)*b*2**(1/2))

Z=objective1(p1,p2)
X11,Y11 = np.meshgrid(p1n,p2n)
Z11= objective1(X11,Y11)

X12,Y12 = np.meshgrid(g1n,g2n)
Z12= objective1(X12,Y12)


y1=model.predict(gtestnew)
y1b=y1.reshape(400,)
tri = mtri.Triangulation(g1n, g2n)

avc=scatter_plot_2d(p1,p2,Z,lim_x=(0,10),lim_y=(0,10),log=False,color_map=0)
bvc=scatter_plot_2d(g1,g2,y1,lim_x=(0,10),lim_y=(0,10),log=False,color_map=0)

avc=surface_plot_2d1(X11,Y11,Z11,log=True,color_map=1,lim_x=(0,10),lim_y=(0,10))

fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(g1n, g2n, y1b, triangles=tri.triangles, cmap='jet')
plt.show()

#pointwise_err=np.linalg.norm(y1-test_targets,axis=(1))
#plt.plot(test_data,pointwise_err,"*")
#plt.yscale("log")
#plt.show()
