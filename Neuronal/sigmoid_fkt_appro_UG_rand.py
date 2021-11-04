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
from matplotlib import pyplot
from UniformGrid.Uniform_Grid_1D import UG_1d
x = UG_1d
x1= x.reshape(4096,)

train_data=np.random.choice(x1,3600,False) #zufällig 3600 Punkte aus z ausgewählt
print(x1.shape)
print(train_data.shape)
p=list(set(x1)-set(train_data)) #Differenzmenge gebildet um testmenge zu finden
test_data=np.array(p)

y1 = 1/(1+e**-train_data)
y2 =1/(1+e**-test_data)
#scale_y = MinMaxScaler()
#y = scale_y.fit_transform(y)
train_targets=y1
test_targets=y2

train_data= train_data.reshape(3600,1)
train_targets= train_targets.reshape(3600,1)
test_data= test_data.reshape(496,1)
test_targets= test_targets.reshape(496,1)

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