# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell



import random
from math import *
from keras import models
from keras import layers
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
from UniformGrid.Uniform_Grid_12D import UG_12d

x = UG_12d

x1= np.transpose(x)
#samplefkt f(x)=sigmoid(x)
z=(x1[0,:]**2+x1[1,:]**2+x1[2,:]**2+x1[3,:]**2+x1[4,:]**2+x1[5,:]**2+x1[6,:]**2+x1[7,:]**2+x1[8,:]**2+x1[9,:]**2+x1[10,:]**2+x1[11,:]**2)
z=np.reshape(z, (4096, 1))

a=(x1[0,:]**2)
a=np.reshape(a, (4096, 1))

b=(x1[1,:]**2)
b=np.reshape(b, (4096, 1))

c=(x1[2,:]**2)
c=np.reshape(c, (4096, 1))

d=(x1[3,:]**2)
d=np.reshape(d, (4096, 1))

e=(x1[4,:]**2)
e=np.reshape(e, (4096, 1))

f=(x1[5,:]**2)
f=np.reshape(f, (4096, 1))

g=(x1[6,:]**2)
g=np.reshape(g, (4096, 1))

h=(x1[7,:]**2)
h=np.reshape(h, (4096, 1))

i=(x1[8,:]**2)
i=np.reshape(i, (4096, 1))

j=(x1[9,:]**2)
j=np.reshape(j, (4096, 1))

k=(x1[10,:]**2)
k=np.reshape(k, (4096, 1))

l=(x1[11,:]**2)
l=np.reshape(l, (4096, 1))


y=1+(1/4000)*(a**2+b**2+c**2+d**2+e**2+f**2+g**2+h**2+i**2+j**2+k**2+l**2)-cos(a)*cos((1/2)*b*2**(1/2))*cos((1/3)*c*3**(1/2))*cos((1/4)*d*4**(1/2))*cos((1/5)*e*5**(1/2))*cos((1/6)*f*6**(1/2))*cos((1/7)*g*7**(1/2))*cos((1/8)*h*8**(1/2))*cos((1/9)*i*9**(1/2))*cos((1/10)*j*10**(1/2))*cos((1/11)*k*11**(1/2))*cos((1/12)*l*12**(1/2))
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

train_data= train_data.reshape(3696,12)
train_targets= train_targets.reshape(3696,1)
test_data= test_data.reshape(400,12)
test_targets= test_targets.reshape(400,1)


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu',))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse', metrics=['mae'])
    return model

model= build_model()
model.fit(train_data,train_targets,epochs=370,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)

#x_plot = scale_x.inverse_transform(x)
#y_plot = scale_y.inverse_transform(y)
#yhat = model.predict(x)

#yhat_plot = scale_y.inverse_transform(yhat)
#print('mse: %.7f' % mean_squared_error(y_plot, yhat_plot))

#pyplot.scatter(x,y, label='Actual')
#pyplot.scatter(x,yhat, label='Predicted')
#pyplot.title('Input (x) versus Output (y)')
#pyplot.xlabel('Input Variable (x)')
#pyplot.ylabel('Output Variable (y)')
#pyplot.legend()
#pyplot.show()