import sys, os
import copy
import pickle
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from utils import divide_cases
from error import rmse, meane

from example_problems import MultiDimJump, FuhgP8
from smt.problems import Sphere, LpNorm, Rosenbrock
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS

# Give directory with desired results as argument
title = sys.argv[1]

if not os.path.isdir(title):
    os.mkdir(title)


# LHS Data
with open(f'./{title}/xk.pickle', 'rb') as f:
    xtrainK = pickle.load(f)
with open(f'./{title}/fk.pickle', 'rb') as f:
    ftrainK = pickle.load(f)
with open(f'./{title}/gk.pickle', 'rb') as f:
    gtrainK = pickle.load(f)
with open(f'./{title}/errkrms.pickle', 'rb') as f:
    errkrms = pickle.load(f)
with open(f'./{title}/errkmean.pickle', 'rb') as f:
    errkmean = pickle.load(f)
# Adaptive Data
with open(f'./{title}/modelf.pickle', 'rb') as f:
    modelf = pickle.load(f)
with open(f'./{title}/err0rms.pickle', 'rb') as f:
    err0rms = pickle.load(f)
with open(f'./{title}/err0mean.pickle', 'rb') as f:
    err0mean = pickle.load(f)
with open(f'./{title}/hist.pickle', 'rb') as f:
    hist = pickle.load(f)
with open(f'./{title}/errhrms.pickle', 'rb') as f:
    errhrms = pickle.load(f)
with open(f'./{title}/errhmean.pickle', 'rb') as f:
    errhmean = pickle.load(f)


# Concatenate lists
xk = []
fk = []
ekr = []
ekm = []
mf = []
e0r = []
e0m = []
hi = []
ehr = []
ehm = []

nprocs = len(modelf)
for i in range(nprocs):
    xk = xk + xtrainK[i][:]
    fk = fk + ftrainK[i][:]
    ekr = ekr + errkrms[i][:]
    ekm = ekm + errkmean[i][:]
    mf = mf + modelf[i][:]
    e0r = e0r + err0rms[i]
    e0m = e0m + err0mean[i]
    hi = hi + hist[i][:]
    ehr = ehr + errhrms[i][:]
    ehm = ehm + errhmean[i][:]

nruns = len(mf)
dim = xk[0].shape[1]

for i in range(nruns):
    ehr[i] = [e0r[i]] + ehr[i] #[errf] #errh
    ehm[i] = [e0m[i]] + ehm[i]



# Plot Error History
iters = len(ehr[0])

samplehist = np.zeros(iters, dtype=int)

for i in range(iters-1):
    samplehist[i] = hi[0][i].ntr
samplehist[iters-1] = mf[0].training_points[None][0][0].shape[0]

# Average out runs
ehrm = np.zeros(iters)
ehmm = np.zeros(iters)
ekrm = np.zeros(iters)
ekmm = np.zeros(iters)

for i in range(nruns):
    ehrm += np.array(ehr[i]).T[0]/nruns
    ehmm += np.array(ehm[i]).T[0]/nruns
    ekrm += np.array(ekr[i]).T[0]/nruns
    ekmm += np.array(ekm[i]).T[0]/nruns

for n in range(nruns):
    plt.loglog(samplehist, ehr[n], "-")
plt.loglog(samplehist, ekrm, 'k--')

plt.grid()
plt.savefig(f"./{title}err_rms_ensemble.png")
plt.clf()


for n in range(nruns):
    plt.loglog(samplehist, ehm[n], "-")
plt.loglog(samplehist, ekmm, 'k--')

plt.grid()
plt.savefig(f"./{title}err_mean_ensemble.png")
plt.clf()

import pdb; pdb.set_trace()

# # Plot points
# if(dim == 2):
