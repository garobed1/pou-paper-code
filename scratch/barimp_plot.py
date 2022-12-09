import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
import math
import importlib
import matplotlib.pyplot as plt

from problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate, POUHessian
from direct_gek import DGEK
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

dir_list = [
    "pou_paper_hess_arctan_1D",
    "pou_paper_hess_fuhgp3_1D",
    "pou_paper_hess_fuhgsh_1D",
    "pou_paper_hess_arctan_2D",
    "pou_paper_hess_rosenbrock_2D",
    "pou_paper_hess_peaks_2D",
    "pou_paper_hess_ishigami_3D",
    "pou_paper_hess_waterflow_8D",
    "pou_paper_hess_wingweight_10D",
    "pou_paper_hess_arctan_12D"
]

nprob = len(dir_list)


# for key in dir_list:
