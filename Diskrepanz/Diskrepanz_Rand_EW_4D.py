# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Diskrepanz_4D import count_range_in_list
x=0
D3=0
c = []

for x in range(0,20):
    rand=np.random.rand(4096, 4)
    i=0
    n=4096
    D2=0
    while i<=n:
        d = i / n
        q=count_range_in_list(rand,0,d)
        d_H = ((q / len(rand)) - (d ** 4))**2
        D2 = D2 + d_H
        i += 1
    Dis = np.sqrt((1 / n) * D2)
    print(Dis)
    D3=D3+Dis
    #print(D3)
    c.append(Dis)
    x+=1
Ew4=D3/20

var=np.var(c)
#print(var)
#print(Ew4)