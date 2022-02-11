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
import tensorflow as tf

import random
from math import *
from keras import models
from keras import layers
from random_korrekt import rand_2d
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
from scatterplot import scatter_plot_2d
from surfaceplot import surface_plot_2d
from surfaceplot2 import surface_plot_2d1
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8
import matplotlib.tri as mtri
from pylab import imshow, show
from timeit import default_timer as timer

x = rand_2d
x1= np.transpose(x)
#samplefkt f(x)=polyfkt
z=(x1[0,:]**2+x1[1,:]**2)
z=np.reshape(z, (4096, 1))
a=(x1[0,:])
a=np.reshape(a, (4096, 1))
b=(x1[1,:])
b=np.reshape(b, (4096, 1))



y=(-20 * exp(-0.2 * sqrt(0.5 * (z)))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b))) + e + 20)
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

score=[]
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu', ))
    model.add(layers.Dense(64, activation='relu', ))
    model.add(layers.Dense(64, activation='relu', ))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse', metrics=['mae'])
    return model
k=1
num_val_samples = len(val_data)
num_epochs=1000
all_mae_histories=[]

model = build_model()
model.summary()
mc_best = tf.keras.callbacks.ModelCheckpoint("ackley_fkt_appro_rand_2Dto1D" + '/best_model', monitor='val_loss', mode='min',
                                                     save_best_only=True, verbose=0)
#history=model.fit(train_data,train_targets,validation_data=(val_data, val_targets), epochs=1000,batch_size=100,verbose=2,callbacks=mc_best)
#history=model.fit(train_data,train_targets,validation_data=(val_data, val_targets), epochs=1000,batch_size=300,verbose=2,callbacks=mc_best)
history=model.fit(train_data,train_targets,validation_data=(val_data, val_targets), epochs=num_epochs,batch_size=64,verbose=0,callbacks=mc_best)

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
smooth_mae_history = smooth_curve(average_mae_history[300:])


plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochen')
plt.ylabel('MAE Validierung')
#plt.show()

model= tf.keras.models.load_model("ackley_fkt_appro_rand_2Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)
gitter=xtrain
gtest=xtest
gtestnew=gtest.transpose()
sortedArr = gtestnew[gtestnew[:,0].argsort()]
ggg=sortedArr.transpose()
#ggg=xtest

p1 = gitter[0,:]
p1n=p1


p2 = gitter[1,:]
p2n=p2


g1 = ggg[0,:]
g1n=g1



g2 = ggg[1,:]
g2n=g2



def objective1(a,b):
 return (-20 * exp(-0.2 * sqrt(0.5 * ((a**2+b**2))))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b))) + np.e + 20)

Z=objective1(p1,p2)
X11,Y11 = np.meshgrid(p1n,p2n)
Z11= objective1(X11,Y11)

X12,Y12 = np.meshgrid(g1n,g2n)
Z12= objective1(X12,Y12)


y1=model.predict(sortedArr)
y1b=y1.reshape(400,)
tri = mtri.Triangulation(g1n, g2n)

#avc=scatter_plot_2d(p1,p2,Z,lim_x=(0,1),lim_y=(0,1),log=False,color_map=0)
bvc=scatter_plot_2d(g1n,g2n,y1,lim_x=(0,1),lim_y=(0,1),log=False,color_map=0,label_x= "Rand_2D", label_y="",title= "Ackleyfunktion 2D")

#avc=surface_plot_2d1(X11,Y11,Z11,log=True,color_map=1,lim_x=(0,1),lim_y=(0,1))

fig2 = plt.figure(figsize=(5.8, 4.7), dpi=400)
ax = fig2.add_subplot(111, projection='3d')
ax.plot_trisurf(g1n, g2n, y1b, triangles=tri.triangles, cmap='jet')
plt.show()


pointwise_err=np.linalg.norm(y1-test_targets,axis=(1))
scatter_plot_2d(g1n,g2n,pointwise_err,lim_x=(0,1),lim_y=(0,1),log=True,color_map=0,label_x= "rand_2D", label_y="",title= "Ackleyfunktion 2D")