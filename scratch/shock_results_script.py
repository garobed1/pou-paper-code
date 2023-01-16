import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
#import matplotlib.pyplot as plt
from infill.refinecriteria import looCV, HessianFit, TEAD
from infill.aniso_criteria import AnisotropicRefine
from infill.taylor_criteria import TaylorRefine, TaylorExploreRefine
from infill.hess_criteria import HessianRefine, POUSSA
from infill.loocv_criteria import POUSFCVT
from infill.aniso_transform import AnisotropicTransform
from infill.getxnew import getxnew, adaptivesampling
from optimization.defaults import DefaultOptOptions
from utils.sutils import divide_cases
from utils.error import rmse, meane
from functions.problem_picker import GetProblem
from functions.shock_problem import ImpingingShock
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate
from smt.sampling_methods import LHS
from scipy.stats import qmc

import mphys_comp.impinge_setup as default_impinge_setup


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



"""
Perform adaptive sampling and estimate error
"""

header = "pou_paper_hess"
path = None
skip_LHS = True
LHS_batch = 7
runs_per_proc = 1

# Problem Conditions
prob  = "shock"    #problem
dim = 2     #problem dimension


# Surrogate Settings
stype = "pouhess"    #surrogate type
rtype =  "hess"
opt = 'L-BFGS-B' #'SLSQP'#
local = False

# rtype =  "pousfcvt"
# opt = 'SLSQP' #for SFCVT constraint
# local = True
corr  = "squar_exp"  #kriging correlation
poly  = "linear"    #kriging regression 
extra = dim           #gek extra points
t0 = [1e-0]
tb = [1e-5, 2e+1]
rscale = 5.5
rho = 10           #POU parameter

# Adaptive Sampling Settings
nt0  = dim*10       #initial design size
ntr = dim*30       #number of points to add
ntot = nt0 + ntr    #total number of points
batch = 1#dim*2        #batch size for refinement, as a percentage of ntr
Nerr = 5000*dim       #number of test points to evaluate the error
pperb = batch
pperbk = int(ntr/LHS_batch)
mstarttype = 2            # 0: No multistart
                          # 1: Start at the best out of a number of samples
                          # 2: Perform multiple optimizations
if(mstarttype == 1):   
    multistart = 50*dim
if(mstarttype == 2):
    multistart = 5*dim

if(pperb == 0):
    pperb = 1

# Refinement Settings
neval = 1+(dim+2)
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = True
bpen = False
obj = "inv"
nscale = 10.0 #1.0 for 2D
nmatch = dim

rc_print = False#False



# Problem Settings
xlimits = np.zeros([dim,2])
xlimits[0,:] = [23., 27.]
xlimits[1,:] = [0.36, 0.51]

problem_settings = default_impinge_setup
problem_settings.aeroOptions['L2Convergence'] = 1e-15
problem_settings.aeroOptions['printIterations'] = False
problem_settings.aeroOptions['printTiming'] = False

trueFunc = ImpingingShock(ndim=dim, input_bounds=xlimits, comm=MPI.COMM_WORLD, )
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')

# Error
xtest = None 
ftest = None
testdata = None


# Adaptive Sampling Conditions
options = DefaultOptOptions
options["local"] = local
options["localswitch"] = True
options["errorcheck"] = testdata
options["multistart"] = mstarttype
options["lmethod"] = opt

# Print Conditions
if rank == 0:
    print("\n")
    print("\n")
    print("Surrogate Type       : ", stype)
    print("Refinement Type      : ", rtype)
    print("Refinement Multistart: ", multistart)
    print("Correlation Function : ", corr)
    print("Regression Function  : ", poly)
    print("GEK Extra Points     : ", extra)
    print("Problem              : ", prob)
    print("Problem Dimension    : ", dim)
    print("Initial Sample Size  : ", nt0)
    print("Refined Points Size  : ", ntr)
    print("Total Points         : ", ntot)
    print("Points Per Iteration : ", int(batch*ntr))
    print("RMSE Size            : ", Nerr)
    print("\n")




    print("Computing Initial Designs for Adaptive Sampling ...")

    # Adaptive Sampling Initial Design
xtrain0 = []
ftrain0 = []
gtrain0 = []
xtrain0 = sampling(nt0)
xtrain0 = comm.bcast(xtrain0, root=0)
ftrain0 = trueFunc(xtrain0)
gtrain0 = np.zeros([nt0,dim])
for i in range(dim):
    gtrain0[:,i:i+1] = trueFunc(xtrain0,i)

ftrain0 = comm.bcast(ftrain0, root=0)
gtrain0 = comm.bcast(gtrain0, root=0)

if rank == 0:
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
elif(stype == "kpls"):
    modelbase = KPLS()
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
else:
    modelbase = KRG()
    modelbase.options.update({"corr":corr})
    modelbase.options.update({"poly":poly})
    modelbase.options.update({"n_start":5})
modelbase.options.update({"print_global":False})

model0 = []

model0 = copy.deepcopy(modelbase)
model0.set_training_values(xtrain0, ftrain0)
if(isinstance(model0, GEKPLS) or isinstance(model0, POUSurrogate)):
    for i in range(dim):
        model0.set_training_derivatives(xtrain0, gtrain0[:,i:i+1], i)
model0.train()



if rank == 0:
    print("Computing Initial Surrogate Error ...")

# Initial Model Error
err0rms = []
err0mean = []
# err0rms = rmse(model0, trueFunc, N=Nerr, xdata=xtest, fdata=ftest)
# err0mean = meane(model0, trueFunc, N=Nerr, xdata=xtest, fdata=ftest)


if rank == 0:
    print("Initial Refinement Criteria ...")

# Initial Refinement Criteria
RC0 = []
if(rtype == "aniso"):
    RC0 = AnisotropicRefine(model0, gtrain0, xlimits, rscale=rscale, nscale=nscale, improve=pperb, neval=neval, hessian=hess, interp=interp, bpen=bpen, objective=obj, multistart=multistart) 
else:
    raise ValueError("Given criteria not valid.")

if rank == 0:
    print("Performing Adaptive Sampling ...")

# Perform Adaptive Sampling
modelf = []
RCF = []
hist = []
errhrms = []
errhmean = []

mf, rF, hf, ef, ef2 = adaptivesampling(trueFunc, model0, RC0, xlimits, ntr, options=options)
modelf = mf
RCF = rF
hist = hf
errhrms = ef
errhmean = ef2

if rank == 0:
    print("\n")
    print("Experiment Complete")

    if(rtype == "aniso"):
        rstring = f'{rtype}{neval}{bpen}{obj}{rscale}r_{nscale}n'
    elif(rtype == "anisotransform"):
        rstring = f'{rtype}{neval}{nmatch}'
    elif(rtype == "tead"):
        rstring = f'{rtype}{neval}'
    else:
        rstring = f'{rtype}'

    title = f'{prob}_{rstring}_{stype}_{corr}_{dim}d_{Nruns}runs_{nt0}to{ntot}pts_{batch}batch_{multistart}mstart_{opt}opt'
    if not os.path.isdir(title):
        os.mkdir(title)

    # Adaptive Data
    with open(f'./{title}/modelf.pickle', 'wb') as f:
        pickle.dump(modelf, f)

    with open(f'./{title}/err0rms.pickle', 'wb') as f:
        pickle.dump(err0rms, f)

    with open(f'./{title}/err0mean.pickle', 'wb') as f:
        pickle.dump(err0mean, f)

    with open(f'./{title}/hist.pickle', 'wb') as f:
        pickle.dump(hist, f)

    with open(f'./{title}/errhrms.pickle', 'wb') as f:
        pickle.dump(errhrms, f)

    with open(f'./{title}/errhmean.pickle', 'wb') as f:
        pickle.dump(errhmean, f)





