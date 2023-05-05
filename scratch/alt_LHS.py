import sys, os
import copy
import pickle
from mpi4py import MPI

import numpy as np
import math
import importlib
import matplotlib.pyplot as plt
from utils.sutils import divide_cases
from utils.error import rmse, meane, full_error

from functions.problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate, POUHessian
from surrogate.direct_gek import DGEK
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
LHS based surrogate error
"""

# Give directory with desired results as argument
delta_x = 1e-7
title = sys.argv[1]
alt_model = ['POU','KRG','GEK']#sys.argv[2]
#impath = title.rsplit('.')
sys.path.append(title)
setmod = importlib.import_module(f'settings')
ssettings = setmod.__dict__

# Give directory for reference data
title2 = sys.argv[2]

with open(f'{title2}/xref.pickle', 'rb') as f:
    xref = pickle.load(f)
with open(f'{title2}/fref.pickle', 'rb') as f:
    fref = pickle.load(f)
with open(f'{title2}/gref.pickle', 'rb') as f:
    gref = pickle.load(f)
    
xref = np.array(xref, dtype=np.float64)
fref = np.array(fref, dtype=np.float64)
gref = np.array(gref, dtype=np.float64)

Nerr = xref.shape[0]

prob = ssettings["prob"]
dim = ssettings["dim"]
#extra = ssettings["extra"]
#corr = ssettings["corr"]
#poly = ssettings["poly"]
#t0 = ssettings["t0"]
#tb = ssettings["tb"]
#rho = ssettings["rho"]
#rscale = ssettings["rscale"]
#neval = ssettings["neval"]
#stype = ssettings["stype"]

trueFunc = GetProblem(prob, dim)#, use_design=ud)
xlimits = trueFunc.xlimits


# LHS Data
with open(f'{title2}/xtrainK.pickle', 'rb') as f:
    xtrainK = pickle.load(f)
with open(f'{title2}/ftrainK.pickle', 'rb') as f:
    ftrainK = pickle.load(f)
with open(f'{title2}/gtrainK.pickle', 'rb') as f:
    gtrainK = pickle.load(f)
#import pdb; pdb.set_trace()

Nruns = size*ssettings["runs_per_proc"]
# Fan out parallel cases
cases = divide_cases(Nruns, size)


# Generate Alternative Surrogate
if(dim > 1):
    modelbase2 = GEKPLS(xlimits=xlimits)
    # modelbase.options.update({"hyper_opt":'TNC'})
    modelbase2.options.update({"theta0":ssettings["t0"]})
    modelbase2.options.update({"theta_bounds":ssettings["tb"]})
    modelbase2.options.update({"n_comp":dim})
    modelbase2.options.update({"extra_points":ssettings["extra"]})
    modelbase2.options.update({"corr":"squar_exp"})#ssettings["corr"]})
    modelbase2.options.update({"poly":ssettings["poly"]})
    modelbase2.options.update({"n_start":5})
    modelbase2.options.update({"delta_x":delta_x})
    if(dim > 2):
        modelbase2.options.update({"zero_out_y":True})
    
# else:
#     modelbase2 = KRG()
#     #modelgek.options.update({"hyper_opt":"TNC"})
#     modelbase2.options.update({"theta0":ssettings["t0"]})
#     modelbase2.options.update({"theta_bounds":ssettings["tb"]})
#     modelbase2.options.update({"corr":"squar_exp"})#ssettings["corr"]})
#     modelbase2.options.update({"poly":ssettings["poly"]})
#     modelbase2.options.update({"n_start":5})
#     modelbase2.options.update({"print_prediction":False})


modelbase1 = KRG()
# modelbase.options.update({"hyper_opt":'TNC'})
modelbase1.options.update({"theta0":ssettings["t0"]})
modelbase1.options.update({"theta_bounds":ssettings["tb"]})
modelbase1.options.update({"corr":"squar_exp"})#ssettings["corr"]})
modelbase1.options.update({"poly":ssettings["poly"]})
modelbase1.options.update({"n_start":5})


modelbase0 = POUHessian(bounds=xlimits, rscale=ssettings['rscale'])
# modelbase.options.update({"hyper_opt":'TNC'})
modelbase0.options.update({"rho":ssettings["rho"]})
modelbase0.options.update({"neval":ssettings["neval"]})

# else:
#     raise ValueError("Given alternative model not valid.")
modelbase0.options.update({"print_global":False})
modelbase1.options.update({"print_global":False})
modelbase2.options.update({"print_global":False})

models = [modelbase0, modelbase1, modelbase2]

slim = 250

slimh0 = slim*10
slimh1 = slim*10
slimh2 = slim*10



samplehistK = [ftrainK[0][i].shape[0] for i in range(len(ftrainK[0]))]
itersk = len(samplehistK)
errk0rms = []
errk1rms = []
errk2rms = []

errors = [errk0rms, errk1rms, errk2rms]

for m in range(itersk):
    for j in range(len(models)):
        models[j].set_training_values(xtrainK[rank][m], ftrainK[rank][m])
        if(isinstance(models[j], GEKPLS) or isinstance(models[j], POUSurrogate) or isinstance(models[j], DGEK) or isinstance(models[j], POUHessian)):
            for i in range(dim):
                models[j].set_training_derivatives(xtrainK[rank][m], gtrainK[rank][m][:,i:i+1], i)
        models[j].train()

        # h = 1e-8
        # x0 = xtrainK[rank][m][0].reshape((1,2))
        # y0 = models[j].predict_values(x0)
        # x1 = copy.deepcopy(xtrainK[rank][m][0].reshape((1,2)))
        # x1[0][0] += h
        # y1 = models[j].predict_values(x1)
        # g = (y1-y0)/h
        
        # # trueFunc(x0)
        # # trueFunc(x0, kx=0)

        # import pdb; pdb.set_trace()


        if(len(ftrainK[rank][m]) > 1000):
            ndir = 150
            xlimits = trueFunc.xlimits
            x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
            y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

            X, Y = np.meshgrid(x, y)
            Za = np.zeros([ndir, ndir])

            for o in range(ndir):
                for p in range(ndir):
                    xi = np.zeros([1,2])
                    xi[0,0] = x[o]
                    xi[0,1] = y[p]
                    Za[p,o] = models[j].predict_values(xi)


            cs = plt.contourf(X, Y, Za, levels = 40)
            plt.colorbar(cs, aspect=20)
            plt.xlabel(r"$x_1$")
            plt.ylabel(r"$x_2$")
            #plt.legend(loc=1)
            nt0 = 20
            plt.plot(xtrainK[0][m][0:nt0,0], xtrainK[0][m][0:nt0,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
            plt.plot(xtrainK[0][m][nt0:,0], xtrainK[0][m][nt0:,1], "o", fillstyle='full', markerfacecolor='r', markeredgecolor='r', label='Adaptive Samples')
            plt.savefig(f"{title}/2d_errcon_LHS.pdf", bbox_inches="tight")
            plt.clf()

        errors[j].append(rmse(models[j], trueFunc, N=Nerr, xdata=xref, fdata=fref))
errors = comm.allgather(errors)

print("HUGE SUCCESS")



if rank == 0:
    print("\n")
    print("Saving Results")

    
    with open(f'{title}/samplehistK.pickle', 'wb') as f:
        pickle.dump(samplehistK, f)

    with open(f'{title}/LHSerrors.pickle', 'wb') as f:
        pickle.dump(errors, f)



    


