import numpy as np
import matplotlib.pyplot as plt
import pougrad

from smt.sampling_methods import LHS
from heaviside import FuhgP8
from smt.surrogate_models import gekpls

dim = 2

trueFunc = FuhgP8()
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits)

ndir = 100
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
X, Y = np.meshgrid(x, y)
Z = np.zeros([ndir, ndir])

for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        Z[i,j] = trueFunc(xi)

#grad 
h = 1e-5
xg = np.zeros([1,2])
xgs = np.zeros([1,2])
xg[0] = [0., 2.0]

fg = trueFunc(xg)
fgs = np.zeros([1,2])
ga = np.zeros([1,2])
for i in range(2):
    ga[0,i] = trueFunc(xg, i)
    xgs[0] = xg[0]
    xgs[0,i] += h
    fgs[0,i] = trueFunc(xgs)

gd = (1./h)*(fgs-fg)

plt.contour(X, Y, Z)
plt.savefig("fuhgp8.png")

import pdb; pdb.set_trace()