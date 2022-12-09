import openmdao.api as om
import sys, os
sys.path.insert(1,"../surrogate")
from sutils import divide_cases
sys.path.insert(1,"../mphys")
from impinge_analysis import Top
from mpi4py import MPI
import numpy as np



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

################################################################################
# OpenMDAO setup
################################################################################
Ncase = 14

prob = om.Problem(comm=MPI.COMM_SELF)
prob.model = Top()
prob.setup(mode='rev')
Y = None
#om.n2(prob, show_browser=False, outfile="mphys_as_adflow_eb_%s_2pt.html")
#prob.set_val("mach", 2.)
#prob.set_val("dv_struct", impinge_setup.structOptions["th"])
#prob.set_val("beta", 7.)
#x = np.linspace(2.5, 3.5, Ncase)
# x = np.linspace(21., 29., Ncase)
# y = np.zeros(Ncase)

# cases = divide_cases(Ncase, size)

# for i in cases[rank]:
#     #prob.set_val("M0", x[i])
#     prob.set_val("shock_angle", x[i])
# #prob.model.approx_totals()
#     prob.run_model()
#     y[i] = prob["test.aero_post.cd_def"]

# Y = comm.allreduce(y)
# totals1 = prob.compute_totals(wrt='rsak')
# prob.model.approx_totals()
prob.run_model()
#totals2 = prob.compute_totals(wrt='shock_angle')
prob.check_totals(wrt='shock_angle')
#prob.check_partials()
#import pdb; pdb.set_trace()
#prob.model.list_outputs()
import pdb; pdb.set_trace()
if MPI.COMM_WORLD.rank == 0:
    print("cd = %.15f" % prob["test.aero_post.cd_def"])
    print(Y)
#     prob.model.