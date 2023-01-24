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
Adaptive sampling-based surrogate error
"""

# Give directory with desired results as argument
title = sys.argv[1]
alt_model = ['POU','KRG','GEK']#sys.argv[2]
#impath = title.rsplit('.')
sys.path.append(title)
setmod = importlib.import_module(f'settings')
ssettings = setmod.__dict__


with open(f'{title}/hist.pickle', 'rb') as f:
    hist0 = pickle.load(f)
with open(f'{title}/hist_30.pickle', 'rb') as f:
    hist1 = pickle.load(f)
with open(f'{title}/hist_55.pickle', 'rb') as f:
    hist2 = pickle.load(f)

nruns = len(hist0) #first dim
iters0 = len(hist0[0][0])
iters1 = len(hist1[0][0])
iters2 = len(hist2[0][0])

full_hist = []
# assuming one run per proc
for i in range(nruns):
    full_hist.append([])
    for j in range(iters0):
        full_hist[i].append(hist0[i][0][j])
    for j in range(iters1):
        full_hist[i].append(hist1[i][0][j])
    for j in range(iters2):
        full_hist[i].append(hist2[i][0][j])


# import pdb; pdb.set_trace()



if rank == 0:
    print("\n")
    print("Saving Results")


    with open(f'{title}/full_hist.pickle', 'wb') as f:
        pickle.dump(full_hist, f)



    


