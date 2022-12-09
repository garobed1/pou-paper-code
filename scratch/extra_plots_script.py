import sys, os
import copy
import pickle
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from refinecriteria import looCV, HessianFit, TEAD
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from sutils import divide_cases
from error import rmse, meane

from ellipse import Ellipse
from example_problems import  Quad2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, MultiDimJumpTwist, FuhgP8, FuhgP9, FuhgP10
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS

# 
dim = 2
prob = "arctan"

# Problem Settings
trueFunc = QuadHadamard(ndim=dim, eigenrate=2.5)
trueFunc2 = QuadHadamard(ndim=dim, eigenrate=0.)

ndir = 13
xlimits = trueFunc.xlimits
xk = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
yk = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
XK, YK = np.meshgrid(xk, yk)

ndir2 = 100
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir2)
y = np.linspace(xlimits[0][0], xlimits[0][1], ndir2)
X, Y = np.meshgrid(x,y)
Z = np.zeros([ndir2, ndir2])
Z2 = np.zeros([ndir2, ndir2])
XJ = np.zeros([ndir, ndir])
YJ = np.zeros([ndir, ndir])
k = 1.4
squeeze = np.array([[k , 1/k],[1/k, k]])
#rot = np.array([[0 , 1],[-1, 0]])
rot = np.eye(2)
for i in range(ndir2):
    for j in range(ndir2):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        Z[i,j] = -trueFunc(xi)
        Z2[i,j] = -trueFunc2(xi)
        
for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = xk[i]
        xi[0,1] = yk[j]
        xit = np.matmul(np.matmul(rot,squeeze), xi.T)
        XJ[i,j] = xit[0]
        YJ[i,j] = xit[1]

# xj = xk + 1.1*ZGX
# yj = yk + 1.1*ZGY
#import pdb; pdb.set_trace()
#XJ, YJ = np.meshgrid(xj,yj)

x1 = -6.5/2.
x2 = -2./2.

# select y-range for zoomed region
y1 = -3./2.
y2 = 4./2.

#plot 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable="box")
plt.rcParams['font.size'] = '11'
plt.plot(XK, YK, "o", fillstyle='full', markerfacecolor='w', markeredgecolor='b')
plt.plot(XJ, YJ, "o", fillstyle='full', markerfacecolor='w', markeredgecolor='r')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xticks([])
plt.yticks([])
#plt.grid()
#plt.clf()

plt.contour(X, Y, Z, [-1., 0.])
#plt.plot(XK, YK, "o", fillstyle='full', markerfacecolor='w', markeredgecolor='b')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xticks([])
plt.yticks([])
plt.xlim([-4.5,4.5])
plt.ylim([-4.5,4.5])
#plt.grid()
plt.savefig(f"./aniso.png", bbox_inches="tight")


#axz = plt.axis([x1, x2, y1, y2])
plt.xlabel("")
plt.ylabel("")
plt.xlim([x1,x2])
plt.ylim([y1,y2])
plt.savefig(f"./zoom.png", bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable="box")
plt.rcParams['font.size'] = '11'
plt.plot(XK, YK, "o", fillstyle='full', markerfacecolor='w', markeredgecolor='b')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xticks([])
plt.yticks([])
#plt.grid()
#plt.clf()

plt.contour(X, Y, Z2, [-2.3, 0.])
#plt.plot(XK, YK, "o", fillstyle='full', markerfacecolor='w', markeredgecolor='b')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xticks([])
plt.yticks([])
plt.xlim([-4.5,4.5])
plt.ylim([-4.5,4.5])
#plt.grid()
plt.savefig(f"./circle.png", bbox_inches="tight")
plt.clf()

Nerr = 1000
xlimits = np.zeros([dim,2])
xlimits[0,:] = [0., 1.]
xlimits[1,:] = [0., 1.]
sampling = LHS(xlimits=xlimits, criterion='m')
x = sampling(Nerr)
plt.plot(x[:,0], x[:,1], "bo")
plt.savefig(f"./manypoints.png", bbox_inches="tight")
plt.clf()

# distance warping plots
trueFunc3 = MultiDimJump(ndim=2, alpha=5.)
xlimits = trueFunc3.xlimits
ndir3 = 100
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir3)
y = np.linspace(xlimits[0][0], xlimits[0][1], ndir3)
X, Y = np.meshgrid(x,y)
Z = np.zeros([ndir2, ndir2])
for i in range(ndir2):
    for j in range(ndir2):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        Z[i,j] = trueFunc3(xi)


ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
Nsamp = 11
sampling = LHS(xlimits=xlimits*0.8, criterion='m')
x = sampling(Nsamp)

circles = []
ellipses = []
rad = np.zeros(Nsamp)
ratio = np.zeros(Nsamp)
grad = np.zeros([Nsamp, 2])
plt.contour(X, Y, Z, levels=10)
plt.plot(x[:,0], x[:,1], "bo")
for i in range(x.shape[0]):
    rad[i] = np.random.rand(1)*0.4 + 0.15
    for j in range(dim):
        grad[i,j] = trueFunc3(np.array([x[i,:]]),j)
    ratio[i] = np.linalg.norm(grad[i])
ratio /= np.max(ratio)*1.2
for i in range(x.shape[0]):
    circles.append(patches.Circle(x[i,:], rad[i], color='m', fill=False, zorder=2))
    ellipses.append(patches.Ellipse(x[i,:], 2*rad[i], 1.5*(rad[i] - rad[i]*ratio[i]), angle=-(180/np.pi)*np.arctan(grad[i,1]/grad[i,0]), color='m', fill=False, zorder=2))
    ax.add_patch(circles[i])
#plt.plot(XK, YK, "o", fillstyle='full', markerfacecolor='w', markeredgecolor='b')

plt.xticks([])
plt.yticks([])
#plt.grid()
plt.savefig(f"./circles.png", bbox_inches="tight")
plt.clf()

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.contour(X, Y, Z, levels=10)
plt.colorbar()
plt.plot(x[:,0], x[:,1], "bo")
for i in range(x.shape[0]):
    ax.add_patch(ellipses[i])
plt.xticks([])
plt.yticks([])
#plt.grid()
plt.savefig(f"./ellipses.png", bbox_inches="tight")
plt.clf()

