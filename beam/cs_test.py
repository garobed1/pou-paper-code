import numpy as np
import openmdao.api as om
import mphys_eb as meb
from beam_solver import EulerBeamSolver

# test complex step derivatives for the beam solver

beam_opts={}
solver_dict={}
Nelem = 10

name = 'cstest'
ndv = Nelem+1
ndof = 3 #bad hacky way to do this
number_of_nodes = Nelem+1
L = 1
E = 100
force = 5*np.ones(Nelem+1)
th = 0.1*np.ones(Nelem+1)

solver_dict['ndv']    = ndv
solver_dict['ndof']   = ndof
solver_dict['number_of_nodes'] = number_of_nodes

beam_opts['name'] = name
beam_opts['Nelem'] = Nelem
beam_opts['L'] = L
beam_opts['E'] = E
beam_opts['force'] = force
beam_opts['Iyy'] = None
beam_opts['th'] = th


# create solver
beam_solver_obj = EulerBeamSolver(beam_opts)

solver_objects = [0, 0]
solver_objects[0] = solver_dict
solver_objects[1] = beam_solver_obj

struct_objects = solver_objects

prob = om.Problem()
prob.model.add_subsystem('ebsolver', meb.EBSolver(struct_objects=struct_objects,
            check_partials=False))

prob.model.add_design_var('ebsolver.dv_struct', lower=0.05*np.ones(ndv), upper=0.15*np.ones(ndv))
prob.model.add_objective('ebsolver.struct_states')

prob.setup(force_alloc_complex=True)

prob.check_partials()