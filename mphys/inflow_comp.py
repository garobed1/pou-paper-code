import openmdao.api as om
import numpy as np



class InflowComp(om.ExplicitComponent):
    """Convert upstream mach number, pressure, and temperature to VelocityX, pressure, and density"""
    def initialize(self):
        self.g = 1.4
        self.Rs = 287.055

    def setup(self):
        self.add_input('mach0', 3.0, desc="upstream mach number")
        self.add_input('P0', 2919.0, desc="upstream pressure")
        self.add_input('T0', 217.0, desc="upstream temperature")

        self.add_output('VelocityX', 0.0, desc="upstream velocity")
        self.add_output('Pressure', 0.0, desc="upstream pressure")
        self.add_output('Density', 0.0, desc="upstream density")

        self.declare_partials('*', '*')


    def compute(self, inputs, outputs):

        M0 = inputs['mach0']
        P0 = inputs['P0']
        T0 = inputs['T0']

        # Density
        r0 = P0/(self.Rs*T0)
        outputs['Density'] = r0

        # Speed of sound
        a = np.sqrt(self.g*P0/r0)

        # Velocity
        outputs['VelocityX'] = M0*a

        # Pressure
        outputs['Pressure'] = P0



    def compute_partials(self, inputs, J):

        M0 = inputs['mach0']
        P0 = inputs['P0']
        T0 = inputs['T0']

        # Density
        r0 = P0/(self.Rs*T0)
        drdp = 1./(self.Rs*T0)
        drdt = (-P0/(self.Rs*T0)**2)*self.Rs
        J['Density', 'P0'] = drdp
        J['Density', 'T0'] = drdt

        # Speed of Sound
        a = np.sqrt(self.g*P0/r0)
        dadp = 0.5*(1./np.sqrt(self.g*P0/r0))*self.g/r0
        dadr = 0.5*(1./np.sqrt(self.g*P0/r0))*-self.g*P0/(r0**2)
        dadpt = dadp + dadr*drdp
        dadtt = dadr*drdt 

        # Velocity
        J['VelocityX', 'mach0'] = a
        J['VelocityX', 'P0'] = M0*dadpt
        J['VelocityX', 'T0'] = M0*dadtt

        # Pressure
        J['Pressure', 'P0'] = 1.0



# prob = om.Problem()
# prob.model.add_subsystem('thing', InflowComp())

# prob.setup()
# prob.check_partials()


