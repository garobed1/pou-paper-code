import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
import importlib
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from sutils import divide_cases
from error import rmse, meane, full_error

from example_problems import Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, FakeShock
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, WingWeight
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate, POUHessian
from direct_gek import DGEK
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
In this plotting script, generate desired alternative surrogate models from
those used in adaptive sampling, using the data obtained from adaptive
sampling, and compare performance.
"""

# Give directory with desired results as argument
title = sys.argv[1]
alt_model = sys.argv[2]
setmod = importlib.import_module(f'{title[:-1]}.settings')
ssettings = setmod.__dict__

if not os.path.isdir(title):
    os.mkdir(title)

prob = title.split("_")[-2]
plt.rcParams['font.size'] = '14'


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
# import pdb; pdb.set_trace()
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

# Concatenate lists
xk = []
fk = []
gk = []
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
    gk = gk + gtrainK[i][:]
    ekr = ekr + errkrms[i][:]
    ekm = ekm + errkmean[i][:]
    mf = mf + modelf[i][:]
    e0r = e0r + err0rms[i]
    e0m = e0m + err0mean[i]
    hi = hi + hist[i][:]
    ehr = ehr + errhrms[i][:]
    ehm = ehm + errhmean[i][:]

nruns = len(mf)
nperr = int(nruns/size)
dim = xk[0].shape[1]



# Problem Settings
alpha = 8.       #arctangent jump strength
if(prob == "arctan"):
    trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
elif(prob == "arctantaper"):
    trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
elif(prob == "rosenbrock"):
    trueFunc = Rosenbrock(ndim=dim)
elif(prob == "peaks"):
    trueFunc = Peaks2D(ndim=dim)
elif(prob == "branin"):
    trueFunc = Branin(ndim=dim)
elif(prob == "sphere"):
    trueFunc = Sphere(ndim=dim)
elif(prob == "fuhgp8"):
    trueFunc = FuhgP8(ndim=dim)
elif(prob == "fuhgp9"):
    trueFunc = FuhgP9(ndim=dim)
elif(prob == "fuhgp10"):
    trueFunc = FuhgP10(ndim=dim)
elif(prob == "waterflow"):
    trueFunc = WaterFlow(ndim=dim)
elif(prob == "weldedbeam"):
    trueFunc = WeldedBeam(ndim=dim)
elif(prob == "robotarm"):
    trueFunc = RobotArm(ndim=dim)
elif(prob == "cantilever"):
    trueFunc = CantileverBeam(ndim=dim)
elif(prob == "hadamard"):
    trueFunc = QuadHadamard(ndim=dim)
elif(prob == "lpnorm"):
    trueFunc = LpNorm(ndim=dim)
elif(prob == "wingweight"):
    trueFunc = WingWeight(ndim=dim)
elif(prob == "fakeshock"):
    trueFunc = FakeShock(ndim=dim)
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits

# Get the original testing data
testdata = None
Nerr = 5000
sampling = LHS(xlimits=xlimits, criterion='m')

# Error
xtest = None 
ftest = None
testdata = None
if rank == 0:
    xtest = sampling(Nerr)
    ftest = trueFunc(xtest)

xtest = comm.bcast(xtest, root=0)
ftest = comm.bcast(ftest, root=0)

# Generate Alternative Surrogate
if(alt_model == "gekpls"):
    modelbase = GEKPLS(xlimits=xlimits)
    # modelbase.options.update({"hyper_opt":'TNC'})
    # modelbase.options.update({"theta0":t0g})
    # modelbase.options.update({"theta_bounds":tbg})
    modelbase.options.update({"n_comp":dim})
    modelbase.options.update({"extra_points":ssettings["extra"]})
    modelbase.options.update({"corr":ssettings["corr"]})
    modelbase.options.update({"poly":ssettings["poly"]})
    modelbase.options.update({"n_start":5})
elif(alt_model == "dgek"):
    modelbase = DGEK(xlimits=xlimits)
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase.options.update({"corr":ssettings["corr"]})
    modelbase.options.update({"poly":ssettings["poly"]})
    modelbase.options.update({"n_start":5})
    modelbase.options.update({"theta0":ssettings["t0"]})
    modelbase.options.update({"theta_bounds":ssettings["tb"]})
elif(alt_model == "pou"):
    modelbase = POUSurrogate()
    modelbase.options.update({"rho":ssettings["rho"]})
elif(alt_model == "pouhess"):
    modelbase = POUHessian(bounds=xlimits)
    modelbase.options.update({"rho":ssettings["rho"]})
    modelbase.options.update({"neval":ssettings["neval"]})
elif(alt_model == "kpls"):
    modelbase = KPLS()
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase.options.update({"n_comp":dim})
    modelbase.options.update({"corr":ssettings["corr"]})
    modelbase.options.update({"poly":ssettings["poly"]})
    modelbase.options.update({"n_start":5})
elif(alt_model == "kriging"):
    modelbase = KRG()
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase.options.update({"corr":ssettings["corr"]})
    modelbase.options.update({"poly":ssettings["poly"]})
    modelbase.options.update({"n_start":5})
else:
    raise ValueError("Given alternative model not valid.")
# modelbase.options.update({"print_global":False})
modelbase.options.update({"print_training":True})
modelbase.options.update({"print_prediction":True})
modelbase.options.update({"print_problem":True})
modelbase.options.update({"print_solver":True})








for i in range(nruns):
    ehr[i] = [e0r[i]] + ehr[i] #[errf] #errh
    ehm[i] = [e0m[i]] + ehm[i]


# Plot Error History
iters = len(ehr[0])
itersk = len(ekr[0])

samplehist = np.zeros(iters, dtype=int)
samplehistk = np.zeros(itersk, dtype=int)

for i in range(iters-1):
    samplehist[i] = hi[0][i].ntr
samplehist[iters-1] = mf[0].training_points[None][0][0].shape[0]
for i in range(itersk):
    samplehistk[i] = len(xk[i])



# Grab data from the adaptive sample sets
ind_alt = np.linspace(0, iters, itersk, dtype=int)
xa = []
fa = []
ga = []
for k in range(nruns):
    xa.append([])
    fa.append([])
    ga.append([])
    for i in range(itersk-1):
        xa[k].append(hi[k][ind_alt[i]].model.training_points[None][0][0])
        fa[k].append(hi[k][ind_alt[i]].model.training_points[None][0][1])
        ga[k].append(np.zeros_like(hi[k][ind_alt[i]].model.training_points[None][0][0]))
        for j in range(dim):
            ga[k][i][:,j:j+1] = hi[k][ind_alt[i]].model.training_points[None][j+1][1]
    xa[k].append(mf[k].training_points[None][0][0])
    fa[k].append(mf[k].training_points[None][0][1])
    ga[k].append(np.zeros_like(mf[k].training_points[None][0][0]))
    for j in range(dim):
        ga[k][-1][:,j:j+1] = mf[k].training_points[None][j+1][1]

# Train alternative surrogates
ma = [[] for _ in range(nperr)]
ear = np.zeros([nperr, itersk])
eam = np.zeros([nperr, itersk])
eas = np.zeros([nperr, itersk])

for k in range(nperr):
    ind = k + rank*nperr
    for i in range(itersk):
        ma[k].append(copy.deepcopy(modelbase))
        ma[k][i].set_training_values(xa[ind][i], fa[ind][i])
        if(ma[k][i].supports["training_derivatives"]):
            for j in range(dim):
                ma[k][i].set_training_derivatives(xa[ind][i], ga[ind][i][:,j:j+1], j)
        ma[k][i].train()
        ear[k][i], eam[k][i], eas[k][i] = full_error(ma[k][i], trueFunc, N=5000, xdata=xtest, fdata=ftest)

ma = comm.allgather(ma)
ear = comm.allgather(ear)
eam = comm.allgather(eam)
eas = comm.allgather(eas)
ear = np.concatenate(ear[:], axis=0)
eam = np.concatenate(eam[:], axis=0)
eas = np.concatenate(eas[:], axis=0)

# Average out runs
ehrm = np.zeros(iters)
ehmm = np.zeros(iters)
ehsm = np.zeros(iters) 
ekrm = np.zeros(itersk)
ekmm = np.zeros(itersk)
eksm = np.zeros(itersk)
eamm = np.zeros(itersk)
earm = np.zeros(itersk)
easm = np.zeros(itersk)


for i in range(nruns):
    ehrm += np.array(ehr[i]).T[0]/nruns
    ehmm += np.array(ehm[i]).T[0][0]/nruns
    ehsm += np.array(ehm[i]).T[0][1]/nruns
    ekrm += np.array(ekr[i]).T[0]/nruns
    ekmm += np.array(ekm[i]).T[0][0]/nruns
    eksm += np.array(ekm[i]).T[0][1]/nruns
    earm += np.array(ear[i]).T/nruns
    eamm += np.array(eam[i]).T/nruns
    easm += np.array(eas[i]).T/nruns




if rank == 0:
    #NRMSE
    ax = plt.gca()
    plt.loglog(samplehist, ehrm, "b-", label=f'Adaptive')
    plt.loglog(samplehistk, ekrm, 'k-', label='LHS')
    plt.loglog(samplehistk, earm, 'r-', label=f'Adaptive ({alt_model})')
    plt.xlabel("Number of samples")
    plt.ylabel("NRMSE")
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    plt.legend(loc=3)
    plt.savefig(f"./{title}/err_nrmse_ensemble.png", bbox_inches="tight")
    plt.clf()

    ax = plt.gca()
    plt.loglog(samplehist, ehmm, "b-", label='Adaptive' )
    plt.loglog(samplehistk, ekmm, 'k-', label='LHS')
    plt.loglog(samplehistk, eamm, 'r-', label=f'Adaptive ({alt_model})')
    plt.xlabel("Number of samples")
    plt.ylabel("Mean Error")
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')

    plt.legend(loc=3)
    plt.savefig(f"./{title}/err_mean_ensemble.png", bbox_inches="tight")
    plt.clf()

    ax = plt.gca()
    plt.loglog(samplehist, ehsm, "b-", label='Adaptive' )
    plt.loglog(samplehistk, eksm, 'k-', label='LHS')
    plt.loglog(samplehistk, easm, 'r-', label=f'Adaptive ({alt_model})')
    plt.xlabel("Number of samples")
    plt.ylabel(r"$\sigma$ Error")
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')

    plt.legend(loc=3)
    plt.savefig(f"./{title}/err_stdv_ensemble.png", bbox_inches="tight")
    plt.clf()

