import sys, os
import copy
import pickle
from matplotlib.transforms import Bbox
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from infill.refinecriteria import looCV, HessianFit
from infill.aniso_criteria import AnisotropicRefine
from infill.getxnew import getxnew, adaptivesampling
from optimization.defaults import DefaultOptOptions
from utils.sutils import divide_cases
from utils.error import rmse, meane

from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate
import matplotlib as mpl
from smt.sampling_methods import LHS
from functions.problem_picker import GetProblem

from scipy.spatial import KDTree
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

title = "10000_shock_results"
ntot = 10000

with open(f'./{title}/xref.pickle', 'rb') as f:
    xref = pickle.load(f)
with open(f'./{title}/fref.pickle', 'rb') as f:
    fref = pickle.load(f)
with open(f'./{title}/gref.pickle', 'rb') as f:
    gref = pickle.load(f)

start = 20
stop = 80

trueFunc = GetProblem("shock", 2)
xlimits = trueFunc.xlimits

sample_list = np.arange(start, stop, 10)

sampling = LHS(xlimits=xlimits, criterion='m')

tree = KDTree(xref)

xtrainK = []
ftrainK = []
gtrainK = []

for i in sample_list:
    xsample = sampling(i)
    close_dists, close_inds = tree.query(xsample, k=1)

    xtrainK.append(xref[close_inds])
    ftrainK.append(xref[close_inds])
    gtrainK.append(xref[close_inds])

xtrainK = comm.allgather(xtrainK)
ftrainK = comm.allgather(ftrainK)
gtrainK = comm.allgather(gtrainK)


with open(f'./{title}/xtrainK.pickle', 'wb') as f:
    pickle.dump(xtrainK, f)
with open(f'./{title}/ftrainK.pickle', 'wb') as f:
    pickle.dump(ftrainK, f)
with open(f'./{title}/gtrainK.pickle', 'wb') as f:
    pickle.dump(gtrainK, f)