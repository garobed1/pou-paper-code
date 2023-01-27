import imp
import numpy as np
import argparse
from mpi4py import MPI
import sys
sys.path.insert(1,"../beam/")

import openmdao.api as om

from mphys import Multipoint
from mphys_comp.shock_angle_comp import ShockAngleComp
from mphys_comp.inflow_comp import InflowComp

from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.scenario_aerodynamic import ScenarioAerodynamic

# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_adflow import ADflowBuilder
from mphys_eb import EBBuilder
from mphys_onetoone import OTOBuilder
from mphys.mphys_meld import MeldBuilder
#from mphys.mphys_rlt import RltBuilder

from baseclasses import AeroProblem

# from tacs import elements, constitutive, functions

# contains all options, aero, opt, struct, uq, warp
import mphys_comp.impinge_setup as default_impinge_setup

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

class Top(Multipoint):

    def _declare_options(self):
        self.options.declare('problem_settings', default=default_impinge_setup,
                             desc='default settings for the shock impingement problem, including solver settings.')    


    def setup(self):
        self.impinge_setup = self.options["problem_settings"]

        impinge_setup = self.impinge_setup

        opt_options = impinge_setup.optOptions

        ################################################################################
        # ADflow Setup
        ################################################################################
        aero_options = impinge_setup.aeroOptions
        warp_options = impinge_setup.warpOptions
        def_surf = ['symp1','symp2','wall1','wall2','wall3','inflow','far','outflow']
        struct_surf = 'wall2'

        aero_builder = ADflowBuilder(options=aero_options,  scenario="aerodynamic", def_surf=def_surf, struct_surf=struct_surf) #mesh_options=warp_options,
        aero_builder.initialize(self.comm)
        aero_builder.solver.addFunction('cdv','wall2','cd_def')

        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        dvs.add_output("mach", val=impinge_setup.mach)
        dvs.add_output("beta", val=impinge_setup.beta)
        dvs.add_output("P", val=impinge_setup.P)
        dvs.add_output("T", val=impinge_setup.T)
        # dvs.add_output("shock_angle", opt_options["shock_angle"])

        dvs.add_output("M0", 3.0)
        dvs.add_output('P0', 2919.0)
        dvs.add_output('T0', 217.0)

        # # component to determine post-shock flow properties for ADFlow
        # self.add_subsystem("shock", ShockAngleComp())

        # # component to determine pre-shock flow properties for ADFlow
        self.add_subsystem("upstream", InflowComp())

        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        scenario = "test"
        self.mphys_add_scenario(
            scenario,
            ScenarioAerodynamic(
                aero_builder=aero_builder
            ),
            nonlinear_solver,
            linear_solver,
        )
        self.connect("mesh_aero.x_aero0", "test.x_aero")
        

    def configure(self):
        impinge_setup = self.impinge_setup

        # create the aero problem 
        ap = AeroProblem(
            name=impinge_setup.probName,
            mach=impinge_setup.mach,
            alpha =impinge_setup.alpha,
            beta =impinge_setup.beta,
            areaRef = 1.0,
            chordRef = 1.0,
            T = impinge_setup.T, 
            P = impinge_setup.P, 
            evalFuncs=["cdv","cd_def"],
        )    

        #TODO: Need to find a way to do this for the SA constants

        ap.addDV("mach", value=impinge_setup.mach, name="mach")
        ap.addDV("beta", value=impinge_setup.beta, name="beta")
        ap.addDV("P", value=impinge_setup.P, name="pressure")
        ap.addDV("T", value=impinge_setup.T, name="temperature")


        ap.setBCVar("Pressure", impinge_setup.P0, "inflow")
        ap.setBCVar("Density",  impinge_setup.r0, "inflow")
        ap.setBCVar("VelocityX", impinge_setup.VX, "inflow")

        self.test.coupling.mphys_set_ap(ap)
        self.test.aero_post.mphys_set_ap(ap)

        self.connect("mach", "test.coupling.mach")
        self.connect("mach", "test.aero_post.mach")
        self.connect("beta", "test.coupling.beta")
        self.connect("beta", "test.aero_post.beta")
        self.connect("P", "test.coupling.pressure")
        self.connect("P", "test.aero_post.pressure")
        self.connect("T", "test.coupling.temperature")
        self.connect("T", "test.aero_post.temperature")
        

        
        # self.connect("shock_angle", "shock.shock_angle")
        # self.connect("M0", "shock.mach0")
        # self.connect("P0", "shock.P0")
        # self.connect("T0", "shock.T0")
        self.connect("M0", "upstream.mach0")
        self.connect("P0", "upstream.P0")
        self.connect("T0", "upstream.T0")
        # self.connect("upstream.VelocityX", "test.coupling.aero.VelocityX")
        # self.connect("upstream.Density", "test.coupling.aero.Density")
        # self.connect("upstream.Pressure", "test.coupling.aero.Pressure")
        self.connect("upstream.VelocityX", "test.aero_post.VelocityX")
        self.connect("upstream.Density", "test.aero_post.Density")
        self.connect("upstream.Pressure", "test.aero_post.Pressure")
        # # # ##  EXTREMELY IMPORTANT TO CONNECT AP VARIABLES TO ALL OF THESE
        # # self.connect("shock.mach1", "test.coupling.aero.mach")
        # # self.connect("shock.flow_angle", "test.coupling.aero.beta")
        # # self.connect("shock.T1", "test.coupling.aero.temp")
        # # self.connect("shock.P1", "test.coupling.aero.pressure")
        # self.connect("shock.mach1", "test.aero_post.mach")
        # self.connect("shock.flow_angle", "test.aero_post.beta")
        # self.connect("shock.T1", "test.aero_post.temperature")
        # self.connect("shock.P1", "test.aero_post.pressure")


        self.add_design_var("mach", lower=2.0, upper=2.5)
        self.add_design_var("beta")
        self.add_design_var("P")
        self.add_design_var("T")
        self.add_design_var("M0")
        # self.add_objective("test.aero_post.cdv")
        # self.add_design_var("shock_angle")
        
        self.add_objective("test.aero_post.cd_def")



if __name__ == '__main__':

    ################################################################################
    # OpenMDAO setup
    ################################################################################
    problem_settings = default_impinge_setup
    problem_settings.aeroOptions['L2Convergence'] = 1e-15
    problem_settings.aeroOptions['printIterations'] = False
    problem_settings.aeroOptions['printTiming'] = True

    aeroGridFile = f'../meshes/imp_mphys_73_73_25.cgns'
    problem_settings.aeroOptions['gridFile'] = aeroGridFile


    prob = om.Problem()
    prob.model = Top(problem_settings=problem_settings)
    prob.setup(mode='rev')
    om.n2(prob, show_browser=False, outfile="mphys_as_adflow_eb_%s_2pt.html")
    # prob.set_val("mach", default_impinge_setup.mach)
    # prob.set_val("beta", 10.0)
    prob.run_model()
    prob.check_partials()
    # prob.check_totals(step_calc='rel_avg')

    #prob.model.list_outputs()

    if MPI.COMM_WORLD.rank == 0:
        print("cd = %.15f" % prob["test.aero_post.cd_def"])
    #     prob.model.