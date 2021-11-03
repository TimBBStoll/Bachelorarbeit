# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Uniform_Grid_10d import s
from random_korrekt import rand_6d
from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt


import numpy as np

#erzeugt primzahlen von 2 bis 10000
def primes_from_2_to(n):

    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

#erzeugt van_der_Corput Sequenz
def van_der_corput(n_sample, base=2):

    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence

#halton Verteilung
def halton(dim, n_sample):

    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample



x = halton(6,4096)
Halton_6D=halton(6,4096)
print(x.shape)
#x_h = x[:,6]
#y_h = x[:,8]

#plt.scatter(x_h, y_h)
#plt.show()
grid = s
halt = x
rand= rand_6d


#print(" The MSE is ")
#print(mean_squared_error(rand, grid))