# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Halton_korrekt import halton
from Uniform_Grid_10d import UG_10d

def count_range_in_list(li, min, max):
	ctr = 0
	for a,b,c,d,e,f,g,h,i,j in li:
		if min <= a <= max and min <= b <= max and min <= c <= max and min <= d <= max and min <= e <= max and min <= f <= max and min <= g <= max and min <= h <= max and min <= i <= max and min <= j <= max:
			ctr += 1
	return ctr

a=0
b=1
c=0.0
d=0.5

n=1024
Dim=10

i=1
DN=0
D2=0
x=0

rand = np.random.rand(n,Dim)
while i<=n:
            d=i/n
            q = count_range_in_list(rand, c, d)
            d_H = ((q / len(rand)) - (d ** Dim))**2
            print(d_H)
            D2=D2+d_H #add Diskrepanz zum n-Cube davor
            #Di=d_H+DN
            DN=d_H
            #print(Di)
            i+=1
Dis = np.sqrt((1 / n) * D2)
#Dis2 = np.sqrt((1 / n) * Di)
print(Dis)
#print(Dis2)
