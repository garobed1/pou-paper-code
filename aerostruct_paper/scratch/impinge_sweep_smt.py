import sys, os
sys.path.insert(1,"../surrogate")
from sutils import divide_cases
sys.path.insert(1,"../mphys")
from impinge_analysis import Top
from shock_problem import ImpingingShock
from smt.sampling_methods import LHS
from mpi4py import MPI
import pickle
import numpy as np



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Ncase = 70

inputs = ["shock_angle", "rsak"]
dim = len(inputs)
xlimits = np.zeros([dim,2])
xlimits[0,:] = [23., 27.]
xlimits[1,:] = [0.36, 0.51]
sampling = LHS(xlimits=xlimits, criterion='m')
x = sampling(Ncase)
x = comm.bcast(x, root=0)
#if rank == 0:
title = f'{Ncase}_shock_results'
#    if not os.path.isdir(title):
#        os.mkdir(title)
# with open(f'./{title}/x.pickle', 'rb') as f:
#     x = pickle.load(f)

func = ImpingingShock(ndim=dim, input_bounds=xlimits, inputs=inputs)
sta = 0
sto = 120
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
    title = f'{Ncase}_shock_results'
    with open(f'./{title}/x{sta}to{sto}.pickle', 'wb') as f:
        pickle.dump(x, f)
    with open(f'./{title}/y{sta}to{sto}.pickle', 'wb') as f:
        pickle.dump(y, f)
    with open(f'./{title}/g{sta}to{sto}.pickle', 'wb') as f:
        pickle.dump(g, f)