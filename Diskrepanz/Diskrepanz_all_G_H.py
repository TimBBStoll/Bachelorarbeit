## This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from Diskrepanz_5D import DisG5, DisH5
from Diskrepanz_6D import DisG6, DisH6
from Diskrepanz_7D import DisG7, DisH7
from Diskrepanz_8D import DisG8, DisH8
from Diskrepanz_9D import DisG9, DisH9
from Diskrepanz_10D import DisG10, DisH10

import matplotlib.pyplot as plt
sx_0 = [DisG5, DisH5]
sx_1 = [DisG6, DisH6]
sx_2 = [DisG7, DisH7]
sx_3 = [DisG8, DisH8]
sx_4 = [DisG9, DisH9]
sx_5 = [DisG10, DisH10]
s = np.zeros( [ 6, 2 ] )
s[0] = sx_0
s[1] = sx_1
s[2] = sx_2
s[3] = sx_3
s[4] = sx_4
s[5] = sx_5
print(s)
