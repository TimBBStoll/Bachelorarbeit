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
import numpy as np
from math import *
from keras import models
from keras import layers
import random
from matplotlib import pyplot
from UniformGrid.Uniform_Grid_1D import UG_1d

x = UG_1d

x1= np.transpose(x)
#samplefkt f(x)=polyfkt
y = 30*x**(4)-60*x**(3)+35*x**(2)-4.5*x
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

train_data= train_data.reshape(3696,1)
train_targets= train_targets.reshape(3696,1)
test_data= test_data.reshape(400,1)
test_targets= test_targets.reshape(400,1)


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu',))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss='mse', metrics=['mae'])
    return model

model= build_model()
model.fit(train_data,train_targets,epochs=370,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)


y=model.predict(test_data)

plt.plot(test_data,y,"+")
plt.plot(test_data,test_targets,"*")
plt.show()

pointwise_err=np.linalg.norm(y-test_targets,axis=(1))
plt.plot(test_data,pointwise_err,"*")
plt.yscale("log")
plt.show()

#x_plot = scale_x.inverse_transform(x)
#y_plot = scale_y.inverse_transform(y)
# #yhat = model.predict(x)

#yhat_plot = scale_y.inverse_transform(yhat)
#print('mse: %.7f' % mean_squared_error(y_plot, yhat_plot))

#pyplot.scatter(x,y, label='Actual')
#pyplot.scatter(x,yhat, label='Predicted')
#pyplot.title('Input (x) versus Output (y)')
#pyplot.xlabel('Input Variable (x)')
#pyplot.ylabel('Output Variable (y)')
#pyplot.legend()
#pyplot.show()