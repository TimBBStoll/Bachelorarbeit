# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np
import random
import random2 as rd

import matplotlib.pyplot as plt
import numpy as np

def uniform_grid(n_sample):

    sequence = []
    i=1
    while i < (n_sample+1):
        x = i/(n_sample+1)
        sequence.append(x)
        i= i+1

    return sequence
#print(uniform_grid(9))
x=np.linspace(0,1,11)
y=uniform_grid(9)

x_a= x[1:10]
print(x_a,y)
#X= uniform_grid(20,2)
#Y= uniform_grid(20,2)
#print(X,Y)

plt.scatter(x_a, y)

plt.show()