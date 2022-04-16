import imp
import numpy as np
import argparse
from mpi4py import MPI
import sys
sys.path.insert(1,"../beam/")

import openmdao.api as om

from mphys import Multipoint
from shock_angle_comp import ShockAngleComp
from inflow_comp import InflowComp
from mphys.scenario_aerostructural import ScenarioAeroStructural

# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_adflow import ADflowBuilder
from mphys_eb import EBBuilder
from mphys_onetoone import OTOBuilder
from mphys.mphys_meld import MeldBuilder
#from mphys.mphys_rlt import RltBuilder

from baseclasses import AeroProblem

# from tacs import elements, constitutive, functions

# contains all options, aero, opt, struct, uq, warp
import impinge_setup

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

class Top(Multipoint):
    def setup(self):
        opt_options = impinge_setup.optOptions

        ################################################################################
        # ADflow Setup
        ################################################################################
        aero_options = impinge_setup.aeroOptions
        warp_options = impinge_setup.warpOptions
        def_surf = ['symp1','symp2','wall1','wall2','wall3','inflow','far','outflow']
        struct_surf = 'wall2'

        aero_builder = ADflowBuilder(options=aero_options,  scenario="aerostructural", def_surf=def_surf, struct_surf=struct_surf) #mesh_options=warp_options,
        aero_builder.initialize(self.comm)
        aero_builder.solver.addFunction('cdv','wall2','cd_def')

        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Euler Bernoulli Setup
        ################################################################################
        struct_options = impinge_setup.structOptions

        struct_builder = EBBuilder(struct_options)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Transfer Scheme Setup
        ################################################################################

        self.onetoone = True

        if(self.onetoone):
            ldxfer_builder = OTOBuilder(aero_builder, struct_builder)
            ldxfer_builder.initialize(self.comm)
        else:
            isym = -1
            ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym, n=2)
            ldxfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS Setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        dvs.add_output("shock_angle", opt_options["shock_angle"])
        dvs.add_output("M0", 3.0)
        dvs.add_output('P0', 2919.0)
        dvs.add_output('T0', 217.0)
        dvs.add_output('M1', impinge_setup.mach)
        dvs.add_output('P1', impinge_setup.P)
        dvs.add_output('T1', impinge_setup.T)

        #dvs.add_output("beta", impinge_setup.beta)
        dvs.add_output("dv_struct", struct_options["th"])
        dvs.add_output("rsak", aero_options["SAConsts"][0])

        # component to determine post-shock flow properties for ADFlow
        self.add_subsystem("shock", ShockAngleComp())

        # component to determine pre-shock flow properties for ADFlow
        self.add_subsystem("upstream", InflowComp())

        nonlinear_solver = om.NonlinearBlockGS(maxiter=2, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        scenario = "test"
        self.mphys_add_scenario(
            scenario,
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=ldxfer_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        for discipline in ["aero", "struct"]:
            self.mphys_connect_scenario_coordinate_source("mesh_%s" % discipline, scenario, discipline)

        

    def configure(self):
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
            evalFuncs=["cd_def"],
        )    

        ap.addDV("mach", name="mach1")
        ap.addDV("beta", name="beta")
        ap.addDV("T", name="temp1")
        ap.addDV("P",  name="pressure1")

        # set BC vars that are hard coded in the CGNS mesh, upstream properties
        ap.setBCVar("Pressure", impinge_setup.P0, "inflow")
        ap.setBCVar("Density",  impinge_setup.r0, "inflow")
        ap.setBCVar("VelocityX", impinge_setup.VX, "inflow")

        self.test.coupling.aero.mphys_set_ap(ap)
        self.test.aero_post.mphys_set_ap(ap)

        self.connect("rsak", "test.coupling.aero.rsak")
        self.connect("dv_struct", f"test.dv_struct")
        # self.connect("beta", "test.coupling.aero.beta")
        # self.connect("beta", "test.aero_post.beta")
        self.connect("shock_angle", "shock.shock_angle")
        self.connect("M0", "shock.mach0")
        self.connect("P0", "shock.P0")
        self.connect("T0", "shock.T0")
        self.connect("M0", "upstream.mach0")
        self.connect("P0", "upstream.P0")
        self.connect("T0", "upstream.T0")
        self.connect("upstream.VelocityX", "test.coupling.aero.VelocityX")
        self.connect("upstream.Density", "test.coupling.aero.Density")
        self.connect("upstream.Pressure", "test.coupling.aero.Pressure")
        self.connect("upstream.VelocityX", "test.aero_post.VelocityX")
        self.connect("upstream.Density", "test.aero_post.Density")
        self.connect("upstream.Pressure", "test.aero_post.Pressure")
        # # ##  EXTREMELY IMPORTANT TO CONNECT AP VARIABLES TO ALL OF THESE
        self.connect("shock.mach1", "test.coupling.aero.mach1")
        self.connect("shock.mach1", "test.aero_post.mach1")
        self.connect("shock.flow_angle", "test.coupling.aero.beta")
        self.connect("shock.flow_angle", "test.aero_post.beta")
        self.connect("shock.T1", "test.coupling.aero.temp1")
        self.connect("shock.T1", "test.aero_post.temp1")
        self.connect("shock.P1", "test.coupling.aero.pressure1")
        self.connect("shock.P1", "test.aero_post.pressure1")
        #self.add_design_var("mach", lower=2.0, upper=2.5)
        #self.add_design_var("dv_struct")
        #self.add_design_var("shock_angle")
        #self.add_design_var("beta")
        # self.add_design_var("shock.mach1")
        # self.add_design_var("shock.flow_angle")
        # self.add_design_var("shock.T1")
        # self.add_design_var("shock.P1")
        # self.connect("M1", "test.coupling.aero.mach1")
        # self.connect("M1", "test.aero_post.mach1")
        # self.connect("P1", "test.coupling.aero.pressure1")
        # self.connect("P1", "test.aero_post.pressure1")
        # self.connect("T1", "test.coupling.aero.temp1")
        # self.connect("T1", "test.aero_post.temp1")
        #self.add_design_var("P1")
        # self.add_design_var("rsak")
        # self.add_design_var("M0")
        # self.add_design_var("shock_angle")
        # self.add_objective("test.aero_post.cd_def")


# ################################################################################
# # OpenMDAO setup
# ################################################################################
# prob = om.Problem()
# prob.model = Top()
# prob.setup(mode='rev')
# om.n2(prob, show_browser=False, outfile="mphys_as_adflow_eb_%s_2pt.html")
# #prob.set_val("mach", 2.)
# #prob.set_val("dv_struct", impinge_setup.structOptions["th"])
# #prob.set_val("beta", 7.)
# x = np.linspace(2.5, 3.5, 10)

# y = np.zeros(10)
# for i in range(10):
#     prob.set_val("M0", x[i])
# #prob.model.approx_totals()
#     prob.run_model()
#     y[i] = prob["test.aero_post.cd_def"]
# # totals1 = prob.compute_totals(wrt='rsak')
# # prob.model.approx_totals()
# #totals2 = prob.compute_totals(wrt='M0')
# #prob.check_partials()
# import pdb; pdb.set_trace()
# #prob.model.list_outputs()

# if MPI.COMM_WORLD.rank == 0:
#     print("cd = %.15f" % prob["test.aero_post.cd_def"])
#     print(y)
# #     prob.model.