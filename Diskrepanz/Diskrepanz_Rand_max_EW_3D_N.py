# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Diskrepanz_3D_N import count_range_in_list
x=0
D3=0
c = []

for x in range(0,20):
    rand=np.random.rand(4096, 3)
    i=0
    n=4096
    Dis = 0
    D2 = np.zeros(shape=n)
    while i<=n:
        d = i / n
        q=count_range_in_list(rand,0,d)
        d_H = ((q / len(rand)) - (d ** 3))
        D2[i - 1] = d_H
        i += 1
    Dis = np.amax(np.absolute(D2))
    c.append(Dis)
    x += 1
print(c)
Ew3N = np.mean(c)
var = np.var(c)
print(var)
print(Ew3N)