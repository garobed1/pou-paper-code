import openmdao.api as om
import numpy as np



class ShockAngleComp(om.ExplicitComponent):
    """Influence of shock angle on top-surface properties of the SBLI problem"""
    def initialize(self):
        self.g = 1.4
        self.Rs = 287.055

    def setup(self):
        # This is the only one we can really control, the others are fixed to the mesh and not able to be differentiated by ADFlow as is
        self.add_input('shock_angle', 25., desc="shock angle of origin")
        
        self.add_input('mach0', 3.0, desc="upstream mach number")
        self.add_input('P0', 2919.0, desc="upstream pressure")
        self.add_input('T0', 217.0, desc="upstream temperature")

        self.add_output('flow_angle', 0.0, desc="downstream angle of attack")
        self.add_output('mach1', 0.0, desc="downstream mach number")
        self.add_output('P1', 0.0, desc="downstream pressure")
        self.add_output('T1', 0.0, desc="downstream temperature")

        self.declare_partials('flow_angle', 'shock_angle')
        self.declare_partials('mach1', 'shock_angle')
        self.declare_partials('P1', 'shock_angle')
        self.declare_partials('T1', 'shock_angle')
        
        self.declare_partials('flow_angle', 'mach0')
        self.declare_partials('mach1', 'mach0')
        self.declare_partials('P1', 'mach0')
        self.declare_partials('T1', 'mach0')

    def compute(self, inputs, outputs):

        sa = inputs['shock_angle']*np.pi/180.
        M0 = inputs['mach0']
        P0 = inputs['P0']
        T0 = inputs['T0']
        sinsa = np.sin(sa)
        m2s2 = M0*M0*sinsa*sinsa

        # Flow angle
        work = (self.g + 1)*M0*M0/(2.*(m2s2 - 1.)) - 1.
        work *= np.tan(sa)
        a = np.arctan(1./work)
        outputs['flow_angle'] = a*180./np.pi

        # Downstream Mach number
        work = ((self.g - 1.)*m2s2 + 2.)/(2.*self.g*m2s2 - (self.g -1))
        work /= (np.sin(sa - a)**2)
        outputs['mach1'] = np.sqrt(work)

        # Downstream Pressure
        work = P0*(2*self.g*m2s2 - (self.g - 1))
        work /= (self.g + 1)
        outputs['P1'] = work

        # Downstream Temperature
        work = T0*(2.*self.g*m2s2 - (self.g - 1))*((self.g - 1)*m2s2 + 2.)
        work /= ((self.g+1)**2)*m2s2
        outputs['T1'] = work

    def compute_partials(self, inputs, J):

        sa = inputs['shock_angle']*np.pi/180.
        M0 = inputs['mach0']
        P0 = inputs['P0']
        T0 = inputs['T0']
        sinsa = np.sin(sa)
        m2s2 = M0*M0*sinsa*sinsa

        # precomputed
        dsa_ds = np.pi/180.
        dsinsa_dsa = np.cos(sa)
        dm2s2_dsinsa = 2*M0*M0*sinsa
        dm2s2_ds = dm2s2_dsinsa*dsinsa_dsa*dsa_ds

        dm2s2_dm0 = 2.*M0*sinsa*sinsa

        ## MACH0
        # Flow angle
        w = (self.g + 1)*M0*M0/(2.*(m2s2 - 1.)) - 1.
        dw_dm0 = (2.*(m2s2 - 1.))*(2*M0*(self.g + 1)) - ( (self.g + 1)*M0*M0)*2.*dm2s2_dm0
        dw_dm0 /= (2.*(m2s2 - 1.))*(2.*(m2s2 - 1.))
        w *= np.tan(sa)
        dw_dm0 *= np.tan(sa)
        a = np.arctan(1./w)
        da_dwinv = (1./(1. + (1./w)**2))
        dwinv_dw = -1./(w**2)
        da_dm0 = da_dwinv*dwinv_dw*dw_dm0
        J['flow_angle','mach0'] = da_dm0*180./np.pi

        # Downstream mach number
        wnumer = ((self.g - 1.)*m2s2 + 2.)
        wdenom = (2.*self.g*m2s2 - (self.g -1))
        w = wnumer/wdenom
        dwnumer = (self.g - 1.)*dm2s2_dm0
        dwdenom = 2.*self.g*dm2s2_dm0
        dw_dm0 = (wdenom*dwnumer - wnumer*dwdenom)/(wdenom**2)
        wnumer = w
        dwnumer = dw_dm0
        dwdenom = 2*np.sin(sa - a)*np.cos(sa - a)*(-da_dm0)
        wdenom = (np.sin(sa - a)**2)
        w = wnumer/wdenom
        dw_dm0 = (wdenom*dwnumer - wnumer*dwdenom)/(wdenom**2)
        J['mach1','mach0'] = (0.5/np.sqrt(w))*dw_dm0

        # Downstream pressure
        w = P0*(2*self.g*m2s2 - (self.g - 1))
        dw_dm0 = P0*(2*self.g*dm2s2_dm0)
        w /= (self.g + 1)
        dw_dm0 /= (self.g + 1)
        J['P1','mach0'] = dw_dm0

        # Downstream temperature
        wnumer = T0*(2.*self.g*m2s2 - (self.g - 1))*((self.g - 1)*m2s2 + 2.)
        dwnumer = T0*(2.*self.g*dm2s2_dm0)*((self.g - 1)*m2s2 + 2.)
        dwnumer +=  T0*(2.*self.g*m2s2 - (self.g - 1))* ((self.g - 1)*dm2s2_dm0)
        wdenom = ((self.g+1)**2)*m2s2
        dwdenom = ((self.g+1)**2)*dm2s2_dm0
        dw_dm0 = (wdenom*dwnumer - wnumer*dwdenom)/(wdenom**2)
        J['T1', 'mach0'] = dw_dm0


        ## SHOCK ANGLE
        # Flow angle
        w = (self.g + 1)*M0*M0/(2.*(m2s2 - 1.)) - 1.
        dw = -(self.g + 1)*M0*M0/(2.*(m2s2 - 1.)**2)
        dwc = dw*dm2s2_ds
        dw_ds = dwc*np.tan(sa) + w*dsa_ds*(1./(np.cos(sa)))**2
        w = w*np.tan(sa)
        a = np.arctan(1./w)
        da_dwinv = (1./(1. + (1./w)**2))
        dwinv_dw = -1./(w**2)
        da_ds = da_dwinv*dwinv_dw*dw_ds
        J['flow_angle','shock_angle'] = da_ds*180./np.pi

        # Downstream mach number
        wnumer = ((self.g - 1.)*m2s2 + 2.)
        wdenom = (2.*self.g*m2s2 - (self.g -1))
        w = wnumer/wdenom
        dwnumer = (self.g - 1.)
        dwdenom = 2.*self.g
        dw_dm2s2 = (wdenom*dwnumer - wnumer*dwdenom)/(wdenom**2)
        dw_ds = dw_dm2s2*dm2s2_ds
        wnumer = w
        dwnumer = dw_ds
        dwdenom = 2*np.sin(sa - a)*np.cos(sa - a)*(dsa_ds - da_ds)
        wdenom = (np.sin(sa - a)**2)
        w = wnumer/wdenom
        dw_ds = (wdenom*dwnumer - wnumer*dwdenom)/(wdenom**2)
        J['mach1','shock_angle'] = (0.5/np.sqrt(w))*dw_ds

        # Downstream Pressure
        w = P0*(2*self.g*m2s2 - (self.g - 1))/(self.g + 1)
        dw_dm2s2 = P0*2*self.g/(self.g + 1)
        J['P1', 'shock_angle'] = dw_dm2s2*dm2s2_ds

        # Downstream Temperature
        w1 = T0*(2.*self.g*m2s2 - (self.g - 1))
        w2 = ((self.g - 1)*m2s2 + 2.)
        dw1 = T0*2.*self.g
        dw2 = (self.g - 1)
        wnumer = w1*w2
        dwnumer = (w1*dw2 + w2*dw1)*dm2s2_ds
        wdenom = ((self.g+1)**2)*m2s2
        dwdenom = ((self.g+1)**2)*dm2s2_ds
        J['T1', 'shock_angle'] = (wdenom*dwnumer - wnumer*dwdenom)/(wdenom**2)

if __name__ == '__main__':

    prob = om.Problem()
    prob.model.add_subsystem('thing', ShockAngleComp())
    
    prob.setup()
    prob.check_partials()


