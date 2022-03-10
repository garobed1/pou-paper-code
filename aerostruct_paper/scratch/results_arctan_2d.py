import sys
import copy
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from error import rmse

from example_problems import MultiDimJump
from smt.problems import Sphere, LpNorm
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS

"""
Error estimate for the arctangent jump problem
"""

# Conditions
stype = "gekpls"    #surrogate type
rtype = "hessian" #criteria type
dim = 2           #problem dimension
nt0  = dim*10     #initial design size
ntr = dim*5      #number of points to add
ntot = nt0 + ntr  #total number of points
batch = 0.0       #batch size for refinement, as a percentage of ntr
Nerr = 5000       #number of test points to evaluate the error
pperb = int(batch*ntr)
if(pperb == 0):
    pperb = 1

# Refinement Settings
neval = dim*2
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = False

# Problem Settings
alpha = 8.       #arctangent jump strength
trueFunc = MultiDimJump(ndim=dim, alpha=alpha) #problem Sphere(ndim=dim)#
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m') #initial design scheme

# Print Conditions
print("\n")
print("\n")
print("Surrogate Type       : ", stype)
print("Refinement Type      : ", rtype)
print("Problem              : MultiDimJump")
print("Problem Dimension    : ", dim)
print("Initial Sample Size  : ", nt0)
print("Refined Points Size  : ", ntr)
print("Total Points         : ", ntot)
print("Points Per Iteration : ", int(batch*ntr))
print("RMSE Size            : ", Nerr)
print("\n")

# Error
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
testdata = [xtest, ftest]

# Adaptive Sampling Conditions
options = DefaultOptOptions
options["localswitch"] = True
options["errorcheck"] = testdata

print("Computing Initial Design ...")

# Initial Design
xtrain0 = sampling(nt0)
ftrain0 = trueFunc(xtrain0)
gtrain0 = np.zeros([nt0,dim])
for i in range(dim):
    gtrain0[:,i:i+1] = trueFunc(xtrain0,i)

print("Computing Final Non-Adaptive Design ...")

# Final Design
xtrainK = sampling(ntot)
ftrainK = trueFunc(xtrainK)
gtrainK = np.zeros([ntot,dim])
for i in range(dim):
    gtrainK[:,i:i+1] = trueFunc(xtrainK,i)

print("Training Initial Surrogate ...")

# Initial Design Surrogate
if(stype == "gekpls"):
    model0 = GEKPLS(xlimits=xlimits)
elif(stype == "pou"):
    model0 = POUSurrogate()
# elif(stype == "rbf"):
#     model0 = RBF()
else:
    model0 = KRG()
#gek = POUSurrogate()
#gek.options.update({"rho":rho})
model0.options.update({"n_start":30})
model0.options.update({"print_global":False})
model0.set_training_values(xtrain0, ftrain0)
if(isinstance(model0, GEKPLS) or isinstance(model0, POUSurrogate)):
    for i in range(dim):
        model0.set_training_derivatives(xtrain0, gtrain0[:,i:i+1], i)
model0.train()



print("Computing Initial Surrogate Error ...")

# Initial Model Error
err0 = rmse(model0, trueFunc, N=Nerr, xdata=xtest, fdata=ftest)



print("Computing Final Non-Adaptive Surrogate Error ...")

# Non-Adaptive Model Error
modelK = copy.deepcopy(model0)
modelK.set_training_values(xtrainK, ftrainK)
if(isinstance(model0, GEKPLS) or isinstance(model0, POUSurrogate)):
    for i in range(dim):
        modelK.set_training_derivatives(xtrainK, gtrainK[:,i:i+1], i)
modelK.train()
errk = rmse(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest)



print("Initial Refinement Criteria ...")

# Initial Refinement Criteria
RC0 = HessianFit(model0, gtrain0, improve=pperb, neval=neval, hessian=hess, interp=interp, criteria=criteria, perturb=perturb) #looCV(gek, approx=False)



print("Performing Adaptive Sampling ...")

# Perform Adaptive Sampling
modelF, RCF, hist, errh = adaptivesampling(trueFunc, model0, RC0, xlimits, ntr, options=options)
#modelf.options.update({"print_global":True})
modelF.train()

modelf = modelF
# xf = modelF.training_points[None][0][0]
# ff = modelF.training_points[None][0][1]
# modelf.set_training_values(xf, ff)
# modelf.train()
# errf = rmse(modelf, trueFunc, N=Nerr, xdata=xtest, fdata=ftest)

print("\n")
print("Experiment Complete")



plt.clf()

# Plot Error History
errh = [err0] + errh #[errf] #errh
iters = len(errh)
samplehist = np.zeros(iters, dtype=int)
for i in range(iters):
    samplehist[i] = nt0 + i*pperb

plt.plot(samplehist, errh, "b")

# Plot Non-Adaptive Error
plt.plot([samplehist[0], samplehist[-1]], [errk, errk], "k--")

plt.savefig("arctan_2d_err.png")

plt.clf()

# Plot Training Points
tr = modelf.training_points[None][0][0]
fr = modelf.training_points[None][0][1]
br = hist[0].bads
plt.plot(tr[0:nt0,0], tr[0:nt0,1], "bo")
plt.plot(br[:,0], br[:,1], "go")
plt.plot(tr[nt0:,0], tr[nt0:,1], "ro")

plt.savefig("arctan_2d_pts.png")
import pdb; pdb.set_trace()


# Plot Error Contour
#Contour
ndir = 150
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

X, Y = np.meshgrid(x, y)
Za = np.zeros([ndir, ndir])
Va = np.zeros([ndir, ndir])
V0 = np.zeros([ndir, ndir])
Zk = np.zeros([ndir, ndir])
Vk = np.zeros([ndir, ndir])
Z0 = np.zeros([ndir, ndir])

for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        Za[i,j] = abs(modelf.predict_values(xi) - trueFunc(xi))
        Zk[i,j] = abs(modelK.predict_values(xi) - trueFunc(xi))
        Z0[i,j] = abs(model0.predict_values(xi) - trueFunc(xi))
        # Va[i,j] = modelf.predict_variances(xi)
        # Vk[i,j] = modelK.predict_variances(xi)
        # V0[i,j] = model0.predict_variances(xi)


cs = plt.contour(X, Y, Za, levels = 15)
plt.colorbar(cs)

plt.savefig("arctan_2d_errcona.png")

plt.clf()

# Plot Non-Adaptive Error
tk = modelK.training_points[None][0][0]
plt.plot(tk[:,0], tk[:,1], "bo")
plt.contour(X, Y, Zk, levels = cs.levels)
plt.colorbar(cs)

plt.savefig("arctan_2d_errconk.png")

plt.clf()
plt.plot(tr[0:nt0,0], tr[0:nt0,1], "bo")
plt.contour(X, Y, Z0, levels = cs.levels)
plt.colorbar(cs)

plt.savefig("arctan_2d_errcon0.png")

