import sys, os
import numpy as np
sys.path.insert(1,"../")
from surrogate.example_problems import FakeShock
import matplotlib.pyplot as plt


dim = 2
trueFunc = FakeShock(ndim=dim)
xlimits = trueFunc.xlimits

#Contour
ndir = 150
xlimits = trueFunc.xlimits
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
X, Y = np.meshgrid(x, y)
TF = np.zeros([ndir, ndir])
for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        TF[j,i] = trueFunc(xi)

cs = plt.contour(X, Y, TF, levels = 40)
plt.colorbar(cs, aspect=20)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
#plt.legend(loc=1)
plt.savefig(f"./fakeshock.png", bbox_inches="tight")
plt.clf()