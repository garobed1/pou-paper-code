import openmdao.api as om
import plate_comp as pc

# Script to run plate optimization
prob = om.Problem()
prob.model.add_subsystem('bump_plate', pc.PlateComponent())

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

# negative one so we maximize the objective
prob.model.add_objective('bump_plate.Cd', scaler=1)

prob.setup()

#prob.set_val('a', .5)
#prob.set_val('Area', 10.0, units='m**2')
#prob.set_val('rho', 1.225, units='kg/m**3')
#prob.set_val('Vu', 10.0, units='m/s')

prob.run_driver()

prob.model.list_inputs(values = False, hierarchical=False)
prob.model.list_outputs(values = False, hierarchical=False)

# minimum value
print(prob['a_disk.Cp'])
print(prob['a'])