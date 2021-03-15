import math
import numpy as np
import openmdao.api as om
import plate_comp as pc
from plate_comp_opts import aeroOptions, warpOptions, optOptions

# Script to run plate optimization
prob = om.Problem()
prob.model.add_subsystem('bump_plate', pc.PlateComponent(), promotes_inputs=['a'])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['debug_print'] = ['desvars','objs','totals']

# design vars and objectives
NV = 2*math.trunc(((1.0 - optOptions['DVFraction'])*optOptions['NX']))
ub = optOptions['DVUpperBound']*np.ones(NV)
lb = np.zeros(NV)
prob.model.add_design_var('a', lower=lb, upper=ub)
prob.model.add_objective('bump_plate.Cd', scaler=1)

prob.setup()

#prob.run_driver()
prob.check_partials()

prob.model.list_inputs(values = False, hierarchical=False)
prob.model.list_outputs(values = False, hierarchical=False)

# minimum value
print(prob['bump_plate.Cd'])
print(prob['a'])