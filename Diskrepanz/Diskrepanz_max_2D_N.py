# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Halton_korrekt import halton
from UniformGrid.Uniform_Grid_2D_N import UG_2dN
def count_range_in_list(li, min, max):
	ctr = 0
	for a,b in li:
		if min <= a <= max and min <= b <= max:
			ctr += 1
	return ctr
Dim=2
n=2**12
rand=np.random.rand(n,Dim)
list1=UG_2dN
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
	d_H1 = 0
	d_H2 = 0
	d_H3 = 0
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
DisG2 = np.amax(np.absolute(D1))
DisH2 = np.amax(np.absolute(D2))
DisR2 = np.amax(np.absolute(D3))
#print(DisG2)
#print(DisH2)
#print(DisR2)