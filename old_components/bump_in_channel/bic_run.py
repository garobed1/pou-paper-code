import math
import os, sys
import time
import numpy as np
import openmdao.api as om
import bic_comp as bc
from plate_comp_opts import aeroOptions, warpOptions, optOptions

# Script to run plate optimization
ooptions = optOptions

# Print options file
fname = ooptions['prob_name']+'.txt'
resfile = open(fname, 'w')
log = open("./plate_comp_opts.py", "r").read()
print(log, file = resfile)

#sys.stdout = open(os.devnull, "w")

prob = om.Problem()
prob.model.add_subsystem('bump_plate', bc.BICComponent())

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['debug_print'] = ['desvars','objs']
prob.driver.options['tol'] = 1e-9
prob.driver.options['maxiter'] = 200

# design vars and objectives
prob.model.add_objective('bump_plate.Cd', scaler=1)

prob.setup()

wc0 = time.perf_counter()
pc0 = time.process_time()

prob.run_model()

wc1 = time.perf_counter()
pc1 = time.process_time()
wct = wc1 - wc0
pct = pc1 - pc0

#sys.stdout = sys.__stdout__

prob.model.list_inputs(values = False, hierarchical=False)
prob.model.list_outputs(values = False, hierarchical=False)

# minimum value
print('WC time = %.15g' % wct, file = resfile)
print('PC time = %.15g' % pct, file = resfile)
print('Cd = %.15g' % prob['bump_plate.Cd'], file = resfile)
