# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# import von TF
# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell



import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import boston_housing
(train_data, train_targets),(test_data,test_targets)=boston_housing.load_data()
mean = train_data.mean(axis=0) #mittelwert
train_data -= mean
std = train_data.std(axis=0) #standartabweichung
train_data /= std
test_data -= mean
test_data /= std

from keras import models
from keras import layers
score=[]
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu',))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse', metrics=['mae'])
    return model
k=4
num_val_samples = len(train_data)//k
num_epochs=100
all_mae_histories=[]


for i in range(k):
    print('Durchlauf ',i)
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data=np.concatenate(
        [train_data[:i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]], axis=0)
    model= build_model()
    history=model.fit(partial_train_data,partial_train_targets,validation_data=(val_data, val_targets), epochs=num_epochs,batch_size=1,verbose=0)
    print(history.history.keys())
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
smooth_mae_history = smooth_curve(average_mae_history[10:])


plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochen')
plt.ylabel('Mittlerer absoluter Fehler Validierung')
plt.show()

model= build_model()
model.fit(train_data,train_targets,epochs=45,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mae_score)