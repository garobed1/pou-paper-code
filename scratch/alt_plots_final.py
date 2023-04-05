import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
import math
import importlib
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from aniso_criteria import AnisotropicRefine
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from sutils import divide_cases
from error import rmse, meane, full_error

from problem_picker import GetProblem
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

Adding the option to "double count" the number of samples for gradient models
"""

# Give directory with desired results as argument
title = sys.argv[1]

fac = 1
if len(sys.argv) > 2:
    fac = sys.argv[2]

alt_model = ['KRG','GEK']#sys.argv[2]
#impath = title.rsplit('.')
sys.path.append(title)
setmod = importlib.import_module(f'settings')
ssettings = setmod.__dict__

if not os.path.isdir(title):
    os.mkdir(title)

prob = title.split("_")[-2]
plt.rcParams['font.size'] = '18'
plt.rc('legend',fontsize=14)



# nruns = len(mf)
# nperr = int(nruns/size)
# dim = xk[0].shape[1]

# # Problem Settings
# trueFunc = GetProblem(prob, dim)
# xlimits = trueFunc.xlimits

# Adaptive Data
with open(f'{title}/samplehist.pickle', 'rb') as f:
    samplehist = pickle.load(f)
with open(f'{title}/samplehistk.pickle', 'rb') as f:
    samplehistk = pickle.load(f)

with open(f'{title}/meanspou.pickle', 'rb') as f:
    meanspou = pickle.load(f)
with open(f'{title}/meanskrg.pickle', 'rb') as f:
    meanskrg = pickle.load(f)
with open(f'{title}/meansgek.pickle', 'rb') as f:
    meansgek = pickle.load(f)
with open(f'{title}/stdvspou.pickle', 'rb') as f:
    stdvspou = pickle.load(f)
with open(f'{title}/stdvskrg.pickle', 'rb') as f:
    stdvskrg = pickle.load(f)
with open(f'{title}/stdvsgek.pickle', 'rb') as f:
    stdvsgek = pickle.load(f)

[ehrm, ehmm, ehsm, ekrm, ekmm, eksm] = meanspou 
[earm1, eamm1, easm1, ehrm1, ehmm1, ehsm1] = meanskrg 
[earm2, eamm2, easm2, ehrm2, ehmm2, ehsm2] = meansgek 
[ehrs, ehms, ehss, ekrs, ekms, ekss] = stdvspou 
[ears1, eams1, eass1, ehrs1, ehms1, ehss1] = stdvskrg 
[ears2, eams2, eass2, ehrs2, ehms2, ehss2] = stdvsgek 
#import pdb; pdb.set_trace()

for i in range(1, ehrm.shape[0]):
    if(ehrm[i] > 100):
        ehrm[i] = ehrm[i-1]
        ehmm[i] = ehmm[i-1]
        ehsm[i] = ehsm[i-1]

if rank == 0:
    #NRMSE
    ax = plt.gca()
    plt.loglog(fac*samplehist, ehrm, "b-", label=f'Adapt (POU)')
    # #plt.fill_between(samplehist, ehrm - ehrs, ehrm + ehrs, color='b', alpha=0.2)
    plt.loglog(fac*samplehistk, ekrm, 'b--', label='LHS (POU)')
    #plt.fill_between(samplehistk, ekrm - ekrs, ekrm + ekrs, color='b', alpha=0.1)
    plt.loglog(samplehistk, earm1, 'g-', label=f'Adapt ({alt_model[0]})')
    #plt.fill_between(samplehistk, earm1 - ears1, earm1 + ears1, color='g', alpha=0.2)
    plt.loglog(samplehistk, ehrm1, 'g--',  label=f'LHS ({alt_model[0]})')
    #plt.fill_between(samplehistk, ehrm1 - ehrs1, ehrm1 + ehrs1, color='g', alpha=0.1)
    plt.loglog(fac*samplehistk, earm2, 'r-', label=f'Adapt ({alt_model[1]})')
    #plt.fill_between(samplehistk, earm2 - ears2, earm2 + ears2, color='r', alpha=0.2)
    plt.loglog(fac*samplehistk, ehrm2, 'r--', label=f'LHS ({alt_model[1]})')
    #plt.fill_between(samplehistk, ehrm2 - ehrs2, ehrm2 + ehrs2, color='r', alpha=0.1)

    if fac > 1.0:
        plt.xlabel("Number of samples")
    else:
        plt.xlabel("Sampling Effort")
    plt.ylabel("NRMSE")
    plt.gca().set_ylim(top=10 ** math.ceil(math.log10(max([ehrm[0], ekrm[0], ehrm1[0],ehrm2[0]]))))
    plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(np.nanmin([ehrm[-1], ekrm[-1], ehrm1[-1],ehrm2[-1]]))))
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 50), labels=np.arange(min(samplehist), max(samplehist), 50) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.ticklabel_format(style='plain', axis='x')
    plt.legend(loc=3)
    plt.savefig(f"{title}/err_nrmse_ensemble_alt.pdf", bbox_inches="tight")
    plt.clf()

    ax = plt.gca()
    plt.loglog(samplehist, ehmm, "b-", label=f'Adapt (POU)')
    # #plt.fill_between(samplehist, ehmm - ehms, ehmm + ehms, color='b', alpha=0.2)
    plt.loglog(samplehistk, ekmm, 'b--', label='LHS (POU)')
    #plt.fill_between(samplehistk, ekmm - ekms, ekmm + ekms, color='b', alpha=0.1)
    plt.loglog(samplehistk, eamm1, 'g-', label=f'Adapt ({alt_model[0]})')
    #plt.fill_between(samplehistk, eamm1 - eams1, eamm1 + eams1, color='g', alpha=0.2)
    plt.loglog(samplehistk, ehmm1, 'g--',  label=f'LHS ({alt_model[0]})')
    #plt.fill_between(samplehistk, ehmm1 - ehms1, ehmm1 + ehms1, color='g', alpha=0.1)
    plt.loglog(samplehistk, eamm2, 'r-', label=f'Adapt ({alt_model[1]})')
    #plt.fill_between(samplehistk, eamm2 - eams2, eamm2 + eams2, color='r', alpha=0.2)
    plt.loglog(samplehistk, ehmm2, 'r--', label=f'LHS ({alt_model[1]})')
    #plt.fill_between(samplehistk, ehmm2 - ehms2, ehmm2 + ehms2, color='r', alpha=0.1)
    plt.xlabel("Number of samples")
    plt.ylabel("Mean Error")
    plt.gca().set_ylim(top=10 ** math.ceil(math.log10(max([ehmm[0], ekmm[0], ehmm1[0],ehmm2[0]]))))
    plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(np.nanmin([ehmm[-1], ekmm[-1], ehmm1[-1],ehmm2[-1]]))))
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.ticklabel_format(style='plain', axis='x')

    plt.legend(loc=3)
    plt.savefig(f"{title}/err_mean_ensemble_alt.pdf", bbox_inches="tight")
    plt.clf()

    ax = plt.gca()
    plt.loglog(samplehist, ehsm, "b-", label=f'Adapt (POU)')
    # #plt.fill_between(samplehist, ehsm - ehss, ehsm + ehss, color='b', alpha=0.2)
    plt.loglog(samplehistk, eksm, 'b--', label='LHS (POU)')
    #plt.fill_between(samplehistk, eksm - ekss, eksm + ekss, color='b', alpha=0.1)
    plt.loglog(samplehistk, easm1, 'g-', label=f'Adapt ({alt_model[0]})')
    #plt.fill_between(samplehistk, easm1 - eass1, easm1 + eass1, color='g', alpha=0.2)
    plt.loglog(samplehistk, ehsm1, 'g--',  label=f'LHS ({alt_model[0]})')
    #plt.fill_between(samplehistk, ehsm1 - ehrs1, ehsm1 + ehss1, color='g', alpha=0.1)
    plt.loglog(samplehistk, easm2, 'r-', label=f'Adapt ({alt_model[1]})')
    #plt.fill_between(samplehistk, easm2 - eass2, easm2 + eass2, color='r', alpha=0.2)
    plt.loglog(samplehistk, ehsm2, 'r--', label=f'LHS ({alt_model[1]})')
    #plt.fill_between(samplehistk, ehsm2 - ehss2, ehsm2 + ehss2, color='r', alpha=0.1)
    plt.xlabel("Number of samples")
    plt.ylabel(r"$\sigma$ Error")
    plt.gca().set_ylim(top=10 ** math.ceil(math.log10(max([ehsm[0], eksm[0], ehsm1[0],ehsm2[0]]))))
    plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(np.nanmin([ehsm[-1], eksm[-1], ehsm1[-1],ehsm2[-1]]))))
    plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
    plt.grid()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.ticklabel_format(style='plain', axis='x')

    plt.legend(loc=3)
    plt.savefig(f"{title}/err_stdv_ensemble_alt.pdf", bbox_inches="tight")
    plt.clf()



