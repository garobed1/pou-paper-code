import math
import numpy as np
import openmdao.api as om
import plate_comp as pc
from plate_comp_opts import aeroOptions, warpOptions, optOptions

# Script to run plate optimization
ooptions = optOptions

# Print options file
log = open("./plate_comp_opts.py", "r").read()
print(log)

prob = om.Problem()
prob.model.add_subsystem('bump_plate', pc.PlateComponent(), promotes_inputs=['a'])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['debug_print'] = ['desvars','objs','totals','nl_cons']
prob.driver.options['tol'] = 1e-6


# design vars and objectives
NV = 2*math.trunc(((1.0 - optOptions['DVFraction'])*optOptions['NX']))
ub = optOptions['DVUpperBound']*np.ones(NV)
lb = optOptions['DVLowerBound']*np.zeros(NV)
prob.model.add_design_var('a', lower=lb, upper=ub)
prob.model.add_objective('bump_plate.Cd', scaler=1)
lbc = ooptions['DCMinThick']
lba = ooptions['DCMinArea']
if ooptions['constrain_opt']:
    if ooptions['use_area_con']:
        prob.model.add_constraint('bump_plate.SA', lower = lba, scaler=1)
    else:
        prob.model.add_constraint('bump_plate.TC', lower = lbc, scaler=1)

prob.model.add_constraint('bump_plate.EQ', equals = 0.0, scaler=1)

prob.setup()

if ooptions['check_partials']:
    prob.check_partials(method = 'fd')
elif ooptions['run_once']:
    prob.run_model()
else:
    prob.run_driver()


prob.model.list_inputs(values = False, hierarchical=False)
prob.model.list_outputs(values = False, hierarchical=False)

# minimum value
print(prob['bump_plate.Cd'])
if ooptions['use_area_con']:
    print(prob['bump_plate.SA'])
else:
    print(prob['bump_plate.TC'])
print(prob['a'])