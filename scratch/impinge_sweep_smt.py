import sys, os
from utils.sutils import divide_cases
from functions.shock_problem import ImpingingShock
from functions.example_problems import FakeShock
from smt.sampling_methods import LHS
from mpi4py import MPI
import pickle
import numpy as np

import mphys_comp.impinge_setup as default_impinge_setup



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

args = sys.argv[1:]

Ncase = int(args[0])#12

inputs = ["shock_angle", "rsak"]
dim = len(inputs)
xlimits = np.zeros([dim,2])
xlimits[0,:] = [23., 27.]
xlimits[1,:] = [0.36, 0.51]
# sampling = LHS(xlimits=xlimits, criterion='m')

# x = sampling(Ncase)
title = f'10000_shock_results'
# if rank == 0:
#     if not os.path.isdir(title):
#         os.mkdir(title)
#     if os.path.exists('./{title}/x.npy'):
with open(f'./{title}/x.npy', 'rb') as f:
    x = pickle.load(f)
# x = comm.bcast(x, root=0)

problem_settings = default_impinge_setup
problem_settings.aeroOptions['L2Convergence'] = 1e-15
problem_settings.aeroOptions['printIterations'] = False
problem_settings.aeroOptions['printTiming'] = False

func = ImpingingShock(ndim=dim, input_bounds=xlimits, inputs=inputs, problem_settings=problem_settings)
# func = FakeShock(ndim=dim)

sta = int(args[1])#0
sto = int(args[2])#12
y = func(x[sta:sto])

g = np.zeros_like(x[sta:sto])
for i in range(dim):
    g[:,i:i+1] = func(x[sta:sto],i)


# Y = comm.allreduce(y)
# totals1 = prob.compute_totals(wrt='rsak')
# prob.model.approx_totals()
# prob.run_model()
# #totals2 = prob.compute_totals(wrt='shock_angle')
# prob.check_totals(wrt='shock_angle')
#prob.check_partials()
#import pdb; pdb.set_trace()
#prob.model.list_outputs()
if rank == 0:
    # with open(f'./{title}/x.npy', 'wb') as f:
    #     pickle.dump(x, f)
    with open(f'./{title}/y{sta}to{sto}.npy', 'wb') as f:
        pickle.dump(y, f)
    with open(f'./{title}/g{sta}to{sto}.npy', 'wb') as f:
        pickle.dump(g, f)