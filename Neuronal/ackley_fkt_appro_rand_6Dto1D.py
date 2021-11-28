
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
from random_korrekt import rand_6d

x = rand_6d

x1= np.transpose(x)
#samplefkt f(x)=sigmoid(x)
z=(x1[0,:]**2+x1[1,:]**2+x1[2,:]**2+x1[3,:]**2+x1[4,:]**2+x1[5,:]**2)
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
y=(-20 * exp(-0.2 * sqrt(0.5 * (z**2)))-exp(0.5 * (cos(2 * pi * a)+cos(2 * pi * b)+cos(2 * pi * c)+cos(2 * pi * d)+cos(2 * pi * e)+cos(2 * pi * f))) + e + 20)
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

train_data= train_data.reshape(3696,6)
train_targets= train_targets.reshape(3696,1)
test_data= test_data.reshape(400,6)
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