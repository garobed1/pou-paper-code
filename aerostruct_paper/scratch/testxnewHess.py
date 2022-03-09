import sys
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from pougrad import POUSurrogate
from numpy.linalg import eig


from example_problems import Heaviside, Quad2D, QuadHadamard, MultiDimJump
from smt.problems import Sphere, LpNorm
from smt.surrogate_models import kpls, gekpls
from smt.sampling_methods import LHS

dim = 2
rho = 100

trueFunc = MultiDimJump(ndim=dim, alpha = 15.)#, theta=np.pi/4)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits)

nt0  = 30
dist = 0.01
ntr = 1
nte = 20

# t0 = np.zeros([nt0,dim])
# t0 = np.array([[0.25, 0.75],[0.5, 0.5],[0.75, 0.25]])# 
t0 = sampling(nt0)

g0 = np.zeros([nt0,dim])

# te = sampling(nte)
# fe = np.zeros(nte)

f0 = trueFunc(t0)



for i in range(dim):
    g0[:,i:i+1] = trueFunc(t0,i)
# fe = trueFunc(te)

# for "fake" gradient enhanced kriging in 1 dim
# tk = np.zeros([2*nt0,dim])
# fk = np.zeros([2*nt0,1])
# for i in range(nt0):
#     tk[2*i,[0]] = t0[i,[0]]-dist
#     fk[2*i,[0]] = f0[i,[0]]-dist*g0[i,[0]]
#     tk[2*i+1,[0]] = t0[i,[0]]+dist
#     fk[2*i+1,[0]] = f0[i,[0]]+dist*g0[i,[0]]

#import pdb; pdb.set_trace()
gek = kpls.KPLS()
#gek = gekpls.GEKPLS(xlimits=xlimits)
#gek = POUSurrogate()
#gek.options.update({"rho":rho})
#gek.options.update({"poly":"linear"})
gek.options.update({"print_global":False})
gek.set_training_values(t0, f0)
#gek.set_training_values(np.append(t0, tk, axis=0), np.append(f0, fk, axis=0))
if(isinstance(gek, gekpls.GEKPLS) or isinstance(gek, POUSurrogate)):
    for i in range(dim):
        gek.set_training_derivatives(t0, g0[:,i:i+1], i)
gek.train()

criteria = HessianFit(gek, g0, improve=ntr, neval=5, hessian="neighborhood", interp="honly", criteria="variance") #looCV(gek, approx=False)

#Contour
plt.figure(2)
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
        Z[i,j] = gek.predict_values(xi)#trueFunc(xi)

plt.contour(X, Y, Z, levels = 15)
# import pdb; pdb.set_trace()




options = DefaultOptOptions
# options = None
options["localswitch"] = True
#import pdb; pdb.set_trace()
#gek.name = "GEKPLS"

gek, criteria, hist = adaptivesampling(trueFunc, gek, criteria, xlimits, ntr, options=options)

t0 = gek.training_points[None][0][0]
f0 = gek.training_points[None][0][1]
b0 = hist[0].bads
b1ind = hist[0].bad_nbhd
b1 = t0[b1ind,:][0]
# plt.plot(x, zt, "k-")
# plt.plot(x, zs, "r-")
plt.plot(t0[0:nt0,0], t0[0:nt0,1], "bo")
plt.plot(t0[nt0:,0], t0[nt0:,1], "ro")
plt.plot(b0[:,0], b0[:,1], "go")
plt.plot(b1[:,0], b1[:,1], "mo")

plt.savefig("asplotgek.png")
# plt.figure(1)
# plt.clf()
# plt.plot(x, zlv, "b-")
# plt.savefig("critplotgek.png")

# tloo = criteria.evaluate(te)

