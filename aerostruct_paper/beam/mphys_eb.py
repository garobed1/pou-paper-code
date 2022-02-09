import numpy as np

import openmdao.api as om
from mphys.builder import Builder
from beam_solver import EulerBeamSolver

"""
Wrapper for the beam solver, incorporating as a structural solver in mphys
"""

class EBMesh(om.IndepVarComp):
    """
    Component to read the initial mesh coordinates for the Euler-Bernoulli solver, in the shape of the aero mesh
    """
    def initialize(self):
        self.options.declare('eb_solver', default = None, desc='the beam_solver object', recordable=False)

    def setup(self):
        eb_solver = self.options['eb_solver']
        xpts = eb_solver.getMeshPoints()

        pts = np.zeros(xpts.size*3)
        for i in range(xpts.size):
            pts[3*i] = xpts[i]

        # for the sake of the 2D impingement problem, double up this vector and shift by 1 in the y direction
        pts2 = np.zeros(xpts.size*3)
        for i in range(xpts.size):
            pts2[3*i+1] = 1.0
            pts2[3*i] = xpts[i]
        pts = np.append(pts, pts2)

        self.add_output('x_struct0', distributed=True, val=pts, shape=pts.size, desc='structural node coordinates', tags=['mphys_coordinates'])

class EBSolver(om.ImplicitComponent):
    """
    Component to perform Euler-Bernoulli structural analysis

        - The steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')        

        self.res = None
        self.ans = None
        self.struct_rhs = None
        self.x_save = None

        self.transposed = False
        self.check_partials = False

        self.ndv = None
        self.npnt = None
        self.ndof = None

        self.beam_solver = None
        self.beam_dict = None

        self.old_dvs = None
        self.old_xs = None

    def setup(self):
        self.check_partials = self.options['check_partials']
        self.beam_solver = self.options["struct_objects"][1]
        self.beam_dict = self.options["struct_objects"][0]

        self.ndv =  self.beam_dict['ndv'] #this is just beam thickness for now
        self.npnt = self.beam_dict['ndv']
        self.ndof = self.beam_dict['ndof']

        # create some vectors that we'll need
        self.res        = np.zeros(self.ndof)
        self.Iyy        = np.zeros(self.npnt)
        self.force      = np.zeros(self.npnt)
        self.ans        = np.zeros(2*(self.npnt-2))
        self.struct_rhs = np.zeros(2*(self.npnt-2))

        # OpenMDAO setup
        state_size = len(self.ans)
        # inputs
        self.add_input('dv_struct', distributed=False, shape=self.ndv, desc='tacs design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=True, shape_by_conn=True, desc='structural node coordinates',tags=['mphys_coordinates'])
        self.add_input('struct_force',  distributed=True, shape_by_conn=True, desc='structural load vector', tags=['mphys_coupling'])

        # outputs
        # its important that we set this to zero since this displacement value is used for the first iteration of the aero
        self.add_output('struct_states', distributed=True, shape=state_size, val = np.zeros(state_size), desc='structural state vector', tags=['mphys_coupling'])

        # partials
        self.declare_partials('u_struct',['dv_struct','x_struct0','f_struct'], method='cs')

    def _need_update(self,inputs):

        update = False

        if self.old_dvs is None:
            self.old_dvs = inputs['dv_struct'].copy()
            update =  True

        for dv, dv_old in zip(inputs['dv_struct'],self.old_dvs):
            if np.abs(dv - dv_old) > 0.:#1e-7:
                self.old_dvs = inputs['dv_struct'].copy()
                update =  True

        if self.old_xs is None:
            self.old_xs = inputs['x_struct0'].copy()
            update =  True

        for xs, xs_old in zip(inputs['x_struct0'],self.old_xs):
            if np.abs(xs - xs_old) > 0.:#1e-7:
                self.old_xs = inputs['x_struct0'].copy()
                update =  True

        return update

    def _update_internal(self,inputs,outputs=None):
        if self._need_update(inputs):
            # update Iyy
            self.beam_solver.computeRectMoment(np.array(inputs['dv_struct']))
            # have this function call setIyy internally

            # update force
            # import pdb; pdb.set_trace()
            # self.beam_solver.setLoad(np.array(inputs['struct_force']))

        if outputs is not None:
            ans = self.ans
            ans[:] = outputs['struct_states']

        self.beam_solver.setLoad(np.array(inputs['struct_force']))


    def apply_nonlinear(self, inputs, outputs, residuals):
        
        self._update_internal(inputs,outputs)

        res  = self.beam_solver.getResidual()

        residuals['struct_states'] = res

    def solve_nonlinear(self, inputs, outputs):

        self._update_internal(inputs,outputs)

        ans  = self.beam_solver()
        outputs['struct_states'] = ans

    def solve_linear(self,d_outputs,d_residuals):
        return None
       
    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals):
        return None

class EBGroup(om.Group):
    def initialize(self):
        self.options.declare('solver_objects', recordable=False)
        self.options.declare("aero_coupling", default=False)
        self.options.declare('check_partials')

    def setup(self):
        self.struct_objects = self.options['solver_objects']
        self.aero_coupling = self.options['aero_coupling']
        self.check_partials = self.options['check_partials']

        if self.aero_coupling:
            self.add_subsystem(
                "force",
                EBForce(struct_objects=self.struct_objects),
                promotes_inputs=["f_struct"],
                promotes_outputs=["struct_force"],
            )

        self.add_subsystem('solver', EBSolver(
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['struct_force', 'x_struct0', 'dv_struct'],
            promotes_outputs=['struct_states']
        )

        if self.aero_coupling:
            self.add_subsystem(
                "disp",
                EBDisp(struct_objects=self.struct_objects),
                promotes_inputs=["struct_states"],
                promotes_outputs=["u_struct"],
            )

class EBFuncsGroup(om.Group):
    def initialize(self):
        self.options.declare('solver_objects', recordable=False)
        self.options.declare('check_partials')

    def setup(self):
        self.struct_objects = self.options['solver_objects']
        self.check_partials = self.options['check_partials']

        self.add_subsystem('funcs', EBFunctions(
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['x_struct0', 'u_struct','dv_struct'],
            promotes_outputs=['func_struct']
        )

        self.add_subsystem('mass', EBMass(
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['x_struct0', 'dv_struct'],
            promotes_outputs=['mass'],
        )
        

class EBBuilder(Builder):

    def __init__(self, options, check_partials=False):
        self.options = options
        self.check_partials = check_partials

    def initialize(self, comm):
        solver_dict={}

        name = self.options['name']

        ndv = (self.options['Nelem']+1)

        ndof = 3 #bad hacky way to do this

        number_of_nodes = (self.options['Nelem']+1)

        solver_dict['ndv']    = ndv
        solver_dict['ndof']   = ndof
        solver_dict['number_of_nodes'] = number_of_nodes
        solver_dict['get_funcs'] = self.options['get_funcs']

        # create solver
        beam_solver_obj = EulerBeamSolver(self.options)

        # check if the user provided a load function
        if 'force' in self.options.keys():
            solver_dict['force'] = self.options['force']

        self.solver_objects = [0, 0]
        self.solver_objects[0] = solver_dict
        self.solver_objects[1] = beam_solver_obj

    def get_coupling_group_subsystem(self):
        return EBGroup(solver_objects=self.solver_objects,
                        aero_coupling=True,
                        check_partials=self.check_partials)

    def get_mesh_coordinate_subsystem(self):
        return EBMesh(eb_solver=self.solver_objects[1])

    def get_post_coupling_subsystem(self):
        return EBFuncsGroup(solver_objects=self.solver_objects,
                            check_partials=self.check_partials)

    def get_ndof(self):
        return self.solver_objects[0]['ndof']

    def get_number_of_nodes(self):
        return 2*self.solver_objects[0]['number_of_nodes']

    def get_ndv(self):
        return self.solver_objects[0]['ndv']






class EBForce(om.ExplicitComponent):
    """
    OpenMDAO component that wraps z forces to the beam solver

    """

    def initialize(self):
        self.options.declare('struct_objects', recordable=False)

    def setup(self):
        # self.set_check_partial_options(wrt='*',directional=True)

        self.solver = self.options["struct_objects"][1]
        solver = self.solver

        self.add_input("f_struct", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        local_coord_size = solver.getMeshPoints().size

        self.add_output("struct_force", distributed=True, shape=local_coord_size, val=np.zeros(local_coord_size), tags=["mphys_coupling"])



    def compute(self, inputs, outputs):

        solver = self.solver

        f = inputs["f_struct"]
        
        f_z = np.zeros(int(len(f)/6))
        for i in range(len(f_z)):
            f_z[i] = f[3*i+2]

        # u_struct = np.zeros(2*len(u_z)*3)
        # for i in range(len(u_z)):
        #     u_struct[3*i+2] = u_z[i]
        #     u_struct[len(u_z)*3 + 3*i+2] = u_z[i]
        #import pdb; pdb.set_trace()

        outputs["struct_force"] = f_z


class EBDisp(om.ExplicitComponent):
    """
    OpenMDAO component that wraps z displacements from the beam solver

    """

    def initialize(self):
        self.options.declare('struct_objects', recordable=False)

    def setup(self):
        # self.set_check_partial_options(wrt='*',directional=True)

        self.solver = self.options["struct_objects"][1]
        solver = self.solver

        self.add_input("struct_states", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

        local_coord_size = 2*solver.getMeshPoints().size*3
        #import pdb; pdb.set_trace()
        self.add_output("u_struct", distributed=True, shape=local_coord_size, val=np.zeros(local_coord_size), tags=["mphys_coupling"])



    def compute(self, inputs, outputs):

        solver = self.solver

        # we don't actually know if the solution here is exactly what we want. will need to investigate
        #u = solver.getSolution() #should actually get this from inputs instead
        u = inputs['struct_states']
        u_z = np.zeros(int(len(u)/2 + 2))
        for i in range(1,len(u_z)-1):
            u_z[i] = u[2*i-1]

        u_struct = np.zeros(2*len(u_z)*3)
        for i in range(len(u_z)):
            u_struct[3*i+2] = u_z[i]
            u_struct[len(u_z)*3 + 3*i+2] = u_z[i]
        #import pdb; pdb.set_trace()

        outputs["u_struct"] = u_struct

    # def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

    #     solver = self.solver



class EBFunctions(om.ExplicitComponent):
    """
    Component to compute functions of the Euler-Bernoulli solver
    """
    def initialize(self):
        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')
        
        self.check_partials = False

    def setup(self):

        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        self.ndv = self.struct_objects[0]['ndv']
        self.func_list = self.struct_objects[0]['get_funcs']

        self.beam_solver = self.struct_objects[1]

        # OpenMDAO part of setup
        # TODO move the dv_struct to an external call where we add the DVs
        self.add_input('dv_struct', distributed=False, shape = self.ndv, desc='design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=True, shape_by_conn=True, desc='structural node coordinates',tags=['mphys_coordinates'])
        self.add_input('struct_force', distributed=True, shape_by_conn=True, desc='asdf',tags=['mphys_input'])
        self.add_input('u_struct', distributed=True, shape_by_conn=True, desc='structural state vector', tags=['mphys_coupling'])

        # Remove the mass function from the func list if it is there
        # since it is not dependent on the structural state
        func_no_mass = []
        for func in enumerate(self.func_list):
            if func not in ['beam_mass']:
                func_no_mass.append(func)

        self.func_list = func_no_mass
        if len(self.func_list) > 0:
            self.add_output('func_struct', distributed=False, shape=len(self.func_list), desc='structural function values', tags=['mphys_result'])

            # declare the partials
            #self.declare_partials('f_struct',['dv_struct','x_struct0','u_struct'])

    def _update_internal(self,inputs):
        # update Iyy
        self.beam_solver.computeRectMoment(np.array(inputs['dv_struct']))
        # have this function call setIyy internally

        self.beam_solver.setLoad(np.array(inputs['struct_force']))

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'func_struct' in outputs:
            outputs['func_struct'] = self.beam_solver.evalFunctions(self.func_list)

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if not self.check_partials:
                raise ValueError('EB forward mode requested but not implemented')
        if mode == 'rev':
            # always update internal because same tacs object could be used by multiple scenarios
            # and we need to load this scenario's state back into TACS before doing derivatives
            self._update_internal(inputs)

            if 'func_struct' in d_outputs:
                proc_contribution = d_outputs['func_struct'][:]
            else: # not sure why OM would call this method without func_struct, but here for safety
                proc_contribution = np.zeros(len(self.func_list))
            d_func = self.comm.allreduce(proc_contribution) / self.comm.size

            for ifunc, func in enumerate(self.func_list):
                self.beam_solver.evalFunctions(self.func_list)
                if 'dv_struct' in d_inputs:
                    dvsens = np.zeros(d_inputs['dv_struct'].size)
                    dvsens = self.beam_solver.evalthSens(func)
                    d_inputs['dv_struct'][:] += np.array(dvsens,dtype=float) * d_func[ifunc]

                # if 'x_struct0' in d_inputs:
                #     xpt_sens = self.xpt_sens
                #     xpt_sens_array = xpt_sens.getArray()
                #     self.tacs_assembler.evalXptSens(func, xpt_sens)

                #     d_inputs['x_struct0'][:] += np.array(xpt_sens_array,dtype=float) * d_func[ifunc]

                if 'u_struct' in d_inputs:
                    usens = np.zeros(d_inputs['u_struct'].size)
                    self.beam_solver.evalforceSens(func, usens)
                    d_inputs['u_struct'][:] += np.array(usens,dtype=float) * d_func[ifunc]

class TacsMass(om.ExplicitComponent):
    """
    Component to compute TACS mass
    """
    def initialize(self):
        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')

        self.mass = False

        self.check_partials = False

    def setup(self):

        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        #self.set_check_partial_options(wrt='*',directional=True)

        self.beam_solver = self.struct_objects[1]

        self.ndv  = self.struct_objects[0]['ndv']

        # OpenMDAO part of setup
        self.add_input('dv_struct', distributed=False, shape=self.ndv, desc='design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=True, shape_by_conn=True, desc='structural node coordinates', tags=['mphys_coordinates'])
        self.add_output('mass', val=0.0, distributed=False, desc = 'structural mass', tags=['mphys_result'])
        #self.declare_partials('mass',['dv_struct','x_struct0'])

    def _update_internal(self,inputs):
        # update Iyy
        self.beam_solver.computeRectMoment(np.array(inputs['dv_struct']))
        # have this function call setIyy internally

        self.beam_solver.setLoad(np.array(inputs['struct_force']))

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'mass' in outputs:
            outputs['mass'] = self.beam_solver.evalFunctions(['beam_mass'])

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if not self.check_partials:
                raise ValueError('EB forward mode requested but not implemented')
        if mode == 'rev':
            self._update_internal(inputs)
            if 'mass' in d_outputs:
                func = self.beam_solver.evalFunctions(['beam_mass'])
                if 'dv_struct' in d_inputs:
                    dvsens = np.zeros(d_inputs['dv_struct'].size)
                    self.beam_solver.evalDVSens(func, dvsens)

                    d_inputs['dv_struct'] += np.array(dvsens,dtype=float) * d_outputs['mass']

                # if 'x_struct0' in d_inputs:
                #     xpt_sens = self.xpt_sens
                #     xpt_sens_array = xpt_sens.getArray()
                #     self.tacs_assembler.evalXptSens(func, xpt_sens)
                #     d_inputs['x_struct0'] += np.array(xpt_sens_array,dtype=float) * d_outputs['mass']


class PrescribedLoad(om.ExplicitComponent):
    """
    Prescribe a load to tacs
    """
    def initialize(self):
        self.options.declare('load_function', default = None, desc='function that prescribes the loads', recordable=False)
        self.options.declare('tacs_assembler', recordable=False)

        self.ndof = 0

    def setup(self):
#        self.set_check_partial_options(wrt='*',directional=True)

        # TACS assembler setup
        tacs_assembler = self.options['tacs_assembler']

        # create some TACS vectors so we can see what size they are
        # TODO getting the node sizes should be easier than this...
        xpts  = tacs_assembler.createNodeVec()
        node_size = xpts.getArray().size

        tmp   = tacs_assembler.createVec()
        state_size = tmp.getArray().size
        self.ndof = int(state_size / ( node_size / 3 ))

        # OpenMDAO setup
        self.add_input('x_struct0', distributed=True, shape_by_conn=True, desc='structural node coordinates', tags=['mphys_coordinates'])
        self.add_output('f_struct', distributed=True, shape=state_size,   desc='structural load', tags=['mphys_coupling'])

        #self.declare_partials('f_struct','x_struct0')

    def compute(self,inputs,outputs):
        load_function = self.options['load_function']
        outputs['f_struct'] = load_function(inputs['x_struct0'],self.ndof)