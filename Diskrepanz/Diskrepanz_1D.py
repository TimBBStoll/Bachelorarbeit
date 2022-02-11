# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Halton_korrekt import halton
from UniformGrid.Uniform_Grid_2D import UG_2d
def count_range_in_list(li, min, max):
	ctr = 0
	for a in li:
		if min <= a <= max:
			ctr += 1
	return ctr
Dim=1
n=4096
rand=np.random.rand(n,Dim)
list1=np.linspace(0,1,n)
list2=halton(Dim,n)
list3=rand

z1=len(list1)
z2=len(list2)
z3=len(list3)

i=0
DN=0
D1=0
D2=0
D3=0
while i<=n:
	d=i/n
	q_1 = count_range_in_list(list1, 0, d)
	d_H1 = ((q_1 / z1) - (d ** Dim)) ** 2
	q_2 = count_range_in_list(list2, 0, d)
	d_H2 = ((q_2 / z2) - (d ** Dim)) ** 2
	q_3 = count_range_in_list(list3, 0, d)
	d_H3 = ((q_3 / z3) - (d ** Dim))**2
	#print(d_H)
	D1 = D1 + d_H1
	D2 = D2 + d_H2
	D3 = D3 + d_H3


	i+=1
DisG1 = np.sqrt((1 / n) * D1)
DisH1 = np.sqrt((1 / n) * D2)
DisR1 = np.sqrt((1 / n) * D3)
#print(DisG1)
#print(DisH1)
#print(DisR1)