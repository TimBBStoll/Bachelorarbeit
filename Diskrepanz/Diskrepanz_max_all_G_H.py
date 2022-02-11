## This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Diskrepanz_max_1D import DisG1, DisH1
from Diskrepanz_max_2D_N import DisG2, DisH2
from Diskrepanz_max_3D_N import DisG3, DisH3
from Diskrepanz_max_4D_N import DisG4, DisH4
from Diskrepanz_max_6D_N import DisG6, DisH6
from Diskrepanz_max_12D import DisG12, DisH12
import matplotlib.pyplot as plt
sx_1 = [DisG1, DisH1]
sx_2 = [DisG2, DisH2]
sx_3 = [DisG3, DisH3]
sx_4 = [DisG4, DisH4]
sx_6 = [DisG6, DisH6]
sx_12 = [DisG12, DisH12]
s = np.zeros( [ 6, 2 ] )
s[0] = sx_1
s[1] = sx_2
s[2] = sx_3
s[3] = sx_4
s[4] = sx_6
s[5] = sx_12

print(s)
