# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Halton_korrekt import halton
from UniformGrid.Uniform_Grid_12D import UG_12d

def count_range_in_list(li, min, max):
	ctr = 0
	for a,b,c,d,e,f,g,h,i,j,u,v in li:
		if min <= a <= max and min <= b <= max and min <= c <= max and min <= d <= max and min <= e <= max and min <= f <= max and min <= g <= max and min <= h <= max and min <= i <= max and min <= j <= max and min <= u <= max and min <= v <= max:
			ctr += 1
	return ctr

a=0
b=1


Dim=12
n=2**12
rand=np.random.rand(n,Dim)



list1=UG_12d
list2=halton(Dim,n)
list3=rand

z1=len(list1)
z2=len(list2)
z3=len(list3)

i=0
D1=np.zeros(shape=n)
D2=np.zeros(shape=n)
D3=np.zeros(shape=n)
while i<=n:
	d=i/n
	q_1 = count_range_in_list(list1, 0, d)
	d_H1 = ((q_1 / z1) - (d ** Dim))
	q_2 = count_range_in_list(list2, 0, d)
	d_H2 = ((q_2 / z2) - (d ** Dim))
	q_3 = count_range_in_list(list3, 0, d)
	d_H3 = ((q_3 / z3) - (d ** Dim))
	#print(d_H)
	D1[i - 1] = d_H1
	D2[i - 1] = d_H2
	D3[i - 1] = d_H3


	i+=1
DisG12 = np.amax(np.absolute(D1))
DisH12 = np.amax(np.absolute(D2))
DisR12 = np.amax(np.absolute(D3))
#print(DisG12)
#print(DisH12)
#print(DisR12)