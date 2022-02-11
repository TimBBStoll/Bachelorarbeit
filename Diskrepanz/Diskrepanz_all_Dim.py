# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Diskrepanz_1D import DisG1, DisH1
from Diskrepanz_2D import DisG2, DisH2
from Diskrepanz_3D import DisG3, DisH3
from Diskrepanz_4D import DisG4, DisH4
from Diskrepanz_6D import DisG6, DisH6
from Diskrepanz_12D import DisG12, DisH12
from Diskrepanz_Rand_EW_12D_N import Ew12N
from Diskrepanz_Rand_EW_1D import Ew1
from Diskrepanz_Rand_EW_2D_N import Ew2N
from Diskrepanz_Rand_EW_3D_N import Ew3N
from Diskrepanz_Rand_EW_4D_N import Ew4N
from Diskrepanz_Rand_EW_6D_N import Ew6N
import matplotlib.pyplot as plt
sx_0 = [DisG1, DisH1, Ew1]
sx_1 = [DisG2, DisH2, Ew2N]
sx_2 = [DisG3, DisH3, Ew3N]
sx_3 = [DisG4, DisH4, Ew4N]
sx_4 = [DisG6, DisH6, Ew6N]
sx_5 = [DisG12, DisH12, Ew12N]
s = np.zeros( [ 6, 3 ] )
s[0] = sx_0
s[1] = sx_1
s[2] = sx_2
s[3] = sx_3
s[4] = sx_4
s[5] = sx_5



print(s)