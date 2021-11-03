import matplotlib

import numpy as np

x = np.linspace(-2, 2, 30)
def f(x):
    return -(1/4)*x**4+(1/3)*x**3+(1/2)*x**2-x+1
poly=f(x)
#plt.scatter(x,f(x))
#plt.show()