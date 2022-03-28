import sys
import copy
import pickle
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from error import rmse, meane

from example_problems import MultiDimJump
from smt.problems import Sphere, LpNorm, Rosenbrock
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS

"""
Error estimate for the arctangent jump problem
"""

# Conditions
multistart = 1      #aniso opt multistart
Nruns = 3
stype = "gekpls"    #surrogate type
rtype = "hessian" #criteria type
corr  = "squar_exp" #kriging correlation
poly  = "linear"  #kriging regression 
extra = 1           #gek extra points
dim = 2          #problem dimension
rho = 10            #POU parameter
nt0  = dim*10     #initial design size
ntr = dim*50      #number of points to add
ntot = nt0 + ntr  #total number of points
batch = 0.1   #batch size for refinement, as a percentage of ntr
Nerr = 5000       #number of test points to evaluate the error
pperb = int(batch*ntr)
if(pperb == 0):
    pperb = 1

# Refinement Settings
neval = 1+dim*2
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = True

# Problem Settings
alpha = 8.       #arctangent jump strength
trueFunc = Rosenbrock(ndim=dim)#, alpha=alpha) #problem Sphere(ndim=dim)#
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m') #initial design scheme

# Print Conditions
print("\n")
print("\n")
print("Surrogate Type       : ", stype)
print("Refinement Type      : ", rtype)
print("Correlation Function : ", corr)
print("Regression Function  : ", poly)
print("GEK Extra Points     : ", extra)
print("Problem              : Rosenbrock")
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
xtrain0 = []
ftrain0 = []
gtrain0 = []
for n in range(Nruns):
    xtrain0.append(sampling(nt0))
    ftrain0.append(trueFunc(xtrain0[n]))
    gtrain0.append(np.zeros([nt0,dim]))
    for i in range(dim):
        gtrain0[n][:,i:i+1] = trueFunc(xtrain0[n],i)

print("Computing Final Non-Adaptive Design ...")

# Final Design(s)
xtrainK = []
ftrainK = []
gtrainK = []
samplehistK = np.linspace(nt0, ntot, int((ntot-nt0)/pperb)+1, dtype=int)
for m in range(Nruns):
    xtrainK.append([])
    ftrainK.append([])
    gtrainK.append([])
    for n in range(len(samplehistK)):
        xtrainK[m].append(sampling(nt0+n*int(batch*ntr)))
        ftrainK[m].append(trueFunc(xtrainK[m][n]))
        gtrainK[m].append(np.zeros([nt0+n*int(batch*ntr),dim]))
        for i in range(dim):
            gtrainK[m][n][:,i:i+1] = trueFunc(xtrainK[m][n],i)



print("Training Initial Surrogate ...")

# Initial Design Surrogate
if(stype == "gekpls"):
    modelbase = GEKPLS(xlimits=xlimits)
    modelbase.options.update({"extra_points":extra})
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})

elif(stype == "pou"):
    modelbase = POUSurrogate()
    modelbase.options.update({"rho":rho})
else:
    modelbase = KRG()
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
modelbase.options.update({"print_global":False})

model0 = []
for n in range(Nruns):
    model0.append(copy.deepcopy(modelbase))
    model0[n].set_training_values(xtrain0[n], ftrain0[n])
    if(isinstance(model0[n], GEKPLS) or isinstance(model0[n], POUSurrogate)):
        for i in range(dim):
            model0[n].set_training_derivatives(xtrain0[n], gtrain0[n][:,i:i+1], i)
    model0[n].train()



print("Computing Initial Surrogate Error ...")

# Initial Model Error
err0rms = []
err0mean = []
for n in range(Nruns):
    err0rms.append(rmse(model0[n], trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    err0mean.append(meane(model0[n], trueFunc, N=Nerr, xdata=xtest, fdata=ftest))



print("Computing Final Non-Adaptive Surrogate Error ...")

# Non-Adaptive Model Error
klen = len(samplehistK)
errkrms = np.zeros(klen)
errkmean = np.zeros(klen)
modelK = copy.deepcopy(modelbase)
for m in range(Nruns):
    for n in range(klen):
        modelK.set_training_values(xtrainK[m][n], ftrainK[m][n])
        if(isinstance(modelbase, GEKPLS) or isinstance(modelbase, POUSurrogate)):
            for i in range(dim):
                modelK.set_training_derivatives(xtrainK[m][n], gtrainK[m][n][:,i:i+1], i)
        modelK.train()
        errkrms[n] += (rmse(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))/Nruns
        errkmean[n] += (meane(modelK, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))/Nruns


print("Initial Refinement Criteria ...")

# Initial Refinement Criteria
RC0 = []
for n in range(Nruns):
    RC0.append(AnisotropicRefine(model0[n], gtrain0[n], improve=pperb, neval=neval, hessian=hess, interp=interp, multistart=multistart) )



print("Performing Adaptive Sampling ...")

# Perform Adaptive Sampling
modelf = []
RCF = []
hist = []
errhrms = []
errhmean = []
for n in range(Nruns):
    print("RUN: ", n)
    mf, rF, hf, ef, ef2 = adaptivesampling(trueFunc, model0[n], RC0[n], xlimits, ntr, options=options)
    modelf.append(mf)
    RCF.append(rF)
    hist.append(hf)
    errhrms.append(ef)
    errhmean.append(ef2)

print("\n")
print("Experiment Complete")



plt.clf()



# Plot Error History
for n in range(Nruns):
    errhrms[n] = [err0rms[n]] + errhrms[n] #[errf] #errh
    errhmean[n] = [err0mean[n]] + errhmean[n]

iters = len(errhrms[0])
samplehist = np.zeros(iters, dtype=int)
for i in range(iters):
    samplehist[i] = nt0 + i*pperb

for n in range(Nruns):
    plt.loglog(samplehist, errhrms[n], "-")

plt.grid()

# Plot Non-Adaptive Error
#plt.loglog([samplehist[0], samplehist[-1]], [errkrms, errkrms], "k--")
plt.loglog(samplehistK, errkrms, 'k--')
plt.savefig("rosenbrock_2d_aniso_err_rms_ensemble.png")

plt.clf()


for n in range(Nruns):
    plt.loglog(samplehist, errhmean[n], "-")

plt.grid()

#plt.loglog([samplehist[0], samplehist[-1]], [errkmean, errkmean], "k:")
plt.loglog(samplehistK, errkmean, 'k--')
plt.savefig("rosenbrock_2d_aniso_err_mean_ensemble.png")


plt.clf()
