import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND
import numpy as np

class beamDVComp(om.ExplicitComponent):
    """
    A Bspline interpolation approach for the beam thickness design variables
    """
    def initialize(self):
        self.options.declare("ndv")
        self.options.declare("method")

        self.ndv = None
        self.interp = None

    def setup(self):
        self.ndv = self.options["ndv"]
        self.method = self.options["method"]

        # inputs
        self.add_input('DVS', shape=self.ndv,
                                    distributed=True,
                                    desc='beam thickness design variables, to be interpolated on the beam solver mesh',
                                    tags=['beam_dvs'])

        # outputs
        self.add_output('th', shape_by_conn=True,
                                    distributed=True,
                                    desc='beam thickness on the solver mesh',
                                    tags=['beam_dvs'])

    def setup_partials(self):
        outsize = self._get_var_meta('th', 'size')

        # actually set up the interpolation here
        x = np.linspace(0., 1.0, outsize)
        self.interp = InterpND(method=self.method, num_cp=self.ndv, x_interp=x, delta_x=0.1)

        self.declare_partials('th','DVS')

    def compute(self, inputs, outputs):

        dvs = np.array(inputs['DVS'])

        y = self.interp.evaluate_spline(dvs)

        outputs['th'] = y

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dvs = np.array(inputs['DVS'])

        y, dy_dycp = self.interp.evaluate_spline(dvs, compute_derivative=True)

        partials['th','DVS'] = dy_dycp



# import matplotlib.pyplot as plt

# ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0, 15.0, 10.0, 2.0, 6.0 ])
# ndv = 10
# n = 50
# x = np.linspace(1.0, 12.0, n)

# prob = om.Problem()
 
# comp = beamDVComp(ndv=ndv, method='akima')

# prob.model.add_subsystem('vars', om.IndepVarComp('dv', val=ycp))
# prob.model.add_subsystem('beamdv', comp)
# prob.model.add_subsystem('sink', om.ExecComp('y=x',
#                                             x={'copy_shape':'y'},
#                                             y={'shape':n}))

# prob.model.connect('vars.dv','beamdv.DVS')
# prob.model.connect('beamdv.th','sink.x')

# prob.setup(force_alloc_complex=True)
# prob.run_model()

# x = np.linspace(0,1,n)
# xcp = np.linspace(0,1,ndv)
# y = prob.get_val('sink.y')

# plt.plot(x, y)
# plt.plot(xcp, ycp, 'o')
# plt.savefig('dvinterp.png')

# prob.check_partials()

# print(prob.get_val('sink.y'))