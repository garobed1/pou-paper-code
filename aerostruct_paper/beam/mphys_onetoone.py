from configparser import Error
import numpy as np
import openmdao.api as om
from mphys import Builder

"""
Writing these components to bypass transfer schemes, for grids that match at the interface
"""

class OTODispXfer(om.ExplicitComponent):
    """
    Component to perform one to one displacement transfer
    """
    def initialize(self):
        self.options.declare('struct_ndof')
        self.options.declare('struct_nnodes')
        self.options.declare('aero_nnodes')
        self.options.declare('check_partials')

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None
        self.check_partials = False

    def setup(self):
        self.struct_ndof   = self.options['struct_ndof']
        self.struct_nnodes = self.options['struct_nnodes']
        self.aero_nnodes   = self.options['aero_nnodes']
        self.check_partials= self.options['check_partials']

        if(self.struct_nnodes != self.aero_nnodes):
            Error("Grid Mismatch for One to One transfer!")

        #self.set_check_partial_options(wrt='*',method='cs',directional=True)

        # inputs
        self.add_input('x_struct0', shape_by_conn=True,
                                    distributed=True,
                                    desc='initial structural node coordinates',
                                    tags=['mphys_coordinates'])
        self.add_input('x_aero0',   shape_by_conn=True,
                                    distributed=True,
                                    desc='initial aero surface node coordinates',
                                    tags=['mphys_coordinates'])
        self.add_input('u_struct',  shape_by_conn=True,
                                    distributed=True,
                                    desc='structural node displacements',
                                    tags=['mphys_coupling'])

        # outputs
        self.add_output('u_aero', shape = self.aero_nnodes*3,
                                  distributed=True,
                                  val=np.zeros(self.aero_nnodes*3),
                                  desc='aerodynamic surface displacements',
                                  tags=['mphys_coupling'])

        # partials
        self.declare_partials('u_aero',['u_struct'])

    def compute(self, inputs, outputs):
        x_s0 = np.array(inputs['x_struct0'])
        x_a0 = np.array(inputs['x_aero0'])
        u_a  = np.array(outputs['u_aero'])

        u_s  = np.zeros(self.struct_nnodes*3)
        
        for i in range(3):
            u_s[i::3] = inputs['u_struct'][i::self.struct_ndof]
        
        if(len(u_a) != len(u_s)):
            Error("Grid Mismatch for One to One transfer!")

        u_a = u_s

        outputs['u_aero'] = u_a

    def compute_partials(self, inputs, partials):
        
        # u_a  = np.array(outputs['u_aero'])

        # u_s  = np.zeros(self.struct_nnodes*3)
        
        # for i in range(3):
        #     u_s[i::3] = inputs['u_struct'][i::self.struct_ndof]

        # uatos = np.eye()

        partials['u_aero','u_struct']


class OTOLoadXfer(om.ExplicitComponent):
    """
    Component to perform one to one load transfers
    """
    def initialize(self):
        self.options.declare('struct_ndof')
        self.options.declare('struct_nnodes')
        self.options.declare('aero_nnodes')
        self.options.declare('check_partials')

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None
        self.check_partials = False

    def setup(self):
        self.struct_ndof   = self.options['struct_ndof']
        self.struct_nnodes = self.options['struct_nnodes']
        self.aero_nnodes   = self.options['aero_nnodes']
        self.check_partials= self.options['check_partials']

        #self.set_check_partial_options(wrt='*',method='cs',directional=True)

        if(self.struct_nnodes != self.aero_nnodes):
            Error("Grid Mismatch for One to One transfer!")

        struct_ndof = self.struct_ndof
        struct_nnodes = self.struct_nnodes

        # inputs
        self.add_input('x_struct0', shape_by_conn=True,
                                    distributed=True,
                                    desc='initial structural node coordinates',
                                    tags=['mphys_coordinates'])
        self.add_input('x_aero0', shape_by_conn=True,
                                  distributed=True,
                                  desc='initial aero surface node coordinates',
                                  tags=['mphys_coordinates'])
        self.add_input('u_struct', shape_by_conn=True,
                                   distributed=True,
                                   desc='structural node displacements',
                                   tags=['mphys_coupling'])
        self.add_input('f_aero', shape_by_conn=True,
                                 distributed=True,
                                 desc='aerodynamic force vector',
                                 tags=['mphys_coupling'])

        # outputs
        self.add_output('f_struct', shape = struct_nnodes*struct_ndof,
                                    distributed=True,
                                    desc='structural force vector',
                                    tags=['mphys_coupling'])

        # partials
        self.declare_partials('f_struct',['x_struct0','x_aero0','u_struct','f_aero'])

    def compute(self, inputs, outputs):
        if self.check_partials:
            x_s0 = np.array(inputs['x_struct0'])
            x_a0 = np.array(inputs['x_aero0'])
            self.meld.setStructNodes(x_s0)
            self.meld.setAeroNodes(x_a0)
        f_a =  np.array(inputs['f_aero'])
        f_s = np.zeros(self.struct_nnodes*3)

        u_s  = np.zeros(self.struct_nnodes*3)
        for i in range(3):
            u_s[i::3] = inputs['u_struct'][i::self.struct_ndof]
        u_a = np.zeros(inputs['f_aero'].size)

        if(len(f_a) != len(f_s)):
            Error("Grid Mismatch for One to One transfer!")

        u_a = u_s
        f_s = f_a
        outputs['f_struct'][:] = 0.0
        for i in range(3):
            outputs['f_struct'][i::self.struct_ndof] = f_s[i::3]

    def compute_partials(self, inputs, partials):
    
        partials['f_struct','f_aero']

class OTOBuilder(Builder):
    def __init__(self, aero_builder, struct_builder, check_partials=False):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder
        self.check_partials = check_partials

    def initialize(self, comm):
        self.aero_nnodes = self.aero_builder.get_number_of_nodes()
        self.struct_nnodes = self.struct_builder.get_number_of_nodes()
        self.struct_ndof = self.struct_builder.get_ndof()

    def get_coupling_group_subsystem(self):
        disp_xfer = OTODispXfer(
            struct_ndof=self.struct_ndof,
            struct_nnodes=self.struct_nnodes,
            aero_nnodes=self.aero_nnodes,
            check_partials=self.check_partials
        )

        load_xfer = OTOLoadXfer(
            struct_ndof=self.struct_ndof,
            struct_nnodes=self.struct_nnodes,
            aero_nnodes=self.aero_nnodes,
            check_partials=self.check_partials
        )

        return disp_xfer, load_xfer
