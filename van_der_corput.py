# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np

def van_der_corput(n_sample, base):

    sequence = []
    for i in range(n_sample):
        n_th_number, x = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            x *= base
            n_th_number += remainder / x
        sequence.append(n_th_number)

    return sequence
X= van_der_corput(20,2)
Y= van_der_corput(20,3)
print(X,Y)

plt.scatter(X, Y)

plt.show()