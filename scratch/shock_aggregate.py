import sys, os
import copy
import pickle
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

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

# Give directory with desired results as argument
usetead = False

title = "10000_shock_results"

if not os.path.isdir(title):
    os.mkdir(title)


plt.rcParams['font.size'] = '12'
#indlist = [[0, 96], [96, 192], [192, 288], [288, 384], [384,388], [388,580], [868,1156], [1156,1444],[2884, 3172], [3172, 5000]]

tot = 10000
jump = 250
nfiles = 10000/250
indlist = [[i*jump, (i+1)*jump] for i in range(nfiles)]


### X
with open(f'./{title}/x.pickle', 'rb') as f:
    xreffull = pickle.load(f)

### F, G
fref = None
gref = None
xref = None
total = 0
for key in indlist:
    total += key[1] - key[0]
    xref = np.append(xref, xreffull[key[0]:key[1]])
    with open(f'./{title}/y{key[0]}to{key[1]}.pickle', 'rb') as f:
        fref = np.append(fref, pickle.load(f))
    with open(f'./{title}/g{key[0]}to{key[1]}.pickle', 'rb') as f:
        gref = np.append(gref, pickle.load(f))

xref = xref[1:]
fref = fref[1:]
gref = gref[1:]
xref = np.reshape(xref, [total, 2])
gref = np.reshape(gref, [total, 2])

with open(f'./{title}/xref.pickle', 'wb') as f:
    pickle.dump(xref, f)
with open(f'./{title}/fref.pickle', 'wb') as f:
    pickle.dump(fref, f)
with open(f'./{title}/gref.pickle', 'wb') as f:
    pickle.dump(gref, f)

