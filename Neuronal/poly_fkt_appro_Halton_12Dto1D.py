# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell
# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell



import random
from math import *
from keras import models
import tensorflow as tf
from keras import layers
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
from Halton_korrekt import Halton_12D
import matplotlib.pyplot as plt

x = Halton_12D

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


y=30*(a)**(4)-60*(a)**(3)+35*(a)**(2)-4.5*(a)+30*(b)**(4)-60*(b)**(3)+35*(b)**(2)-4.5*(b)+30*(c)**(4)-60*(c)**(3)+35*(c)**(2)-4.5*(c)+30*(d)**(4)-60*(d)**(3)+35*(d)**(2)-4.5*(d)+30*(e)**(4)-60*(e)**(3)+35*(e)**(2)-4.5*(e)+30*(f)**(4)-60*(f)**(3)+35*(f)**(2)-4.5*(f)+30*(g)**(4)-60*(g)**(3)+35*(g)**(2)-4.5*(g)+30*(h)**(4)-60*(h)**(3)+35*(h)**(2)-4.5*(h)+30*(i)**(4)-60*(i)**(3)+35*(i)**(2)-4.5*(i)+30*(j)**(4)-60*(j)**(3)+35*(j)**(2)-4.5*(j)+30*(k)**(4)-60*(k)**(3)+35*(k)**(2)-4.5*(k)+30*(l)**(4)-60*(l)**(3)+35*(l)**(2)-4.5*(l)
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
mc_best = tf.keras.callbacks.ModelCheckpoint("poly_fkt_appro_Halton_12Dto1D" + '/best_model', monitor='val_loss', mode='min',
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
plt.show()

model= tf.keras.models.load_model("poly_fkt_appro_Halton_12Dto1D/best_model")
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)
