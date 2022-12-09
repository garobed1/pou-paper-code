import math
import os, sys
import time
import numpy as np
import openmdao.api as om
from mpi4py import MPI
import rk4_pend_mlmc as rpm
import rk4_pend_lhs as rpl
from rk4_pend_opts import uqOptions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Script to run plate optimization
uoptions = uqOptions

fname = uoptions['prob_name']+'.txt'
resfile = open(fname, 'w')

# Print options file
if rank == 0:
    log = open("./rk4_pend_opts.py", "r").read()
    print(log, file = resfile)

#sys.stdout = open(os.devnull, "w")
prob = om.Problem()
if uqOptions['mode'] == 'MLMC':
    prob.model.add_subsystem('rk4_pend', rpm.RK4PendCompMLMC())
else:
    prob.model.add_subsystem('rk4_pend', rpl.RK4PendCompLHS())

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
#prob.driver.options['debug_print'] = ['desvars','objs','totals','nl_cons']
prob.driver.options['tol'] = 1e-6

prob.setup()

wc0 = time.perf_counter()
pc0 = time.process_time()

prob.run_model()

wc1 = time.perf_counter()
pc1 = time.process_time()
wct = wc1 - wc0
pct = pc1 - pc0

#sys.stdout = sys.__stdout__

prob.model.list_outputs(values = False, hierarchical=False)

# minimum value
if rank == 0:
    print('WC time = %.15g' % wct, file = resfile)
    print('PC time = %.15g' % pct, file = resfile)
    print('E = %.15g' % prob['rk4_pend.uf_m'], file = resfile)
    print('V = %.15g' % prob['rk4_pend.uf_v'], file = resfile)

    #if uoptions['mode'] == 'MLMC':
    print('N1 = ', prob['rk4_pend.N1'], file = resfile)