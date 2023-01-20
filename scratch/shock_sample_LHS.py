
import pickle
from mpi4py import MPI
import numpy as np


from smt.sampling_methods import LHS
from functions.problem_picker import GetProblem

from scipy.spatial import KDTree


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

sample_list = np.arange(start, stop+1, 10)

sampling = LHS(xlimits=xlimits, criterion='m')

tree = KDTree(xref)

xtrainK = []
ftrainK = []
gtrainK = []

for i in sample_list:
    xsample = sampling(i)
    close_dists, close_inds = tree.query(xsample, k=1)

    xtrainK.append(np.array(xref[close_inds], dtype=np.float64))
    ftrainK.append(np.array(fref[close_inds].reshape((i, 1)), dtype=np.float64))
    gtrainK.append(np.array(gref[close_inds], dtype=np.float64))

xtrainK = comm.allgather(xtrainK)
ftrainK = comm.allgather(ftrainK)
gtrainK = comm.allgather(gtrainK)

if rank == 0:
    with open(f'./{title}/xtrainK.pickle', 'wb') as f:
        pickle.dump(xtrainK, f)
    with open(f'./{title}/ftrainK.pickle', 'wb') as f:
        pickle.dump(ftrainK, f)
    with open(f'./{title}/gtrainK.pickle', 'wb') as f:
        pickle.dump(gtrainK, f)