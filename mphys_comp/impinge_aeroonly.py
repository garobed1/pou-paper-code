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
        self.options.declare('full_free', default=False,
                             desc='apply far field conditions at inflow (everywhere)')    
        self.options.declare('use_inflow_comp', default=True,
                             desc='determine downstream settings ')    
        self.options.declare('use_shock_comp', default=False,
                             desc='determine upstream settings using the oblique shock component')    
        self.options.declare('subsonic', default=False,
                             desc='use subsonic mesh specification instead')    


    def setup(self):
        self.impinge_setup = self.options["problem_settings"]

        impinge_setup = self.impinge_setup

        opt_options = impinge_setup.optOptions

        ################################################################################
        # ADflow Setup
        ################################################################################
        aero_options = impinge_setup.aeroOptions
        warp_options = impinge_setup.warpOptions
        if(self.options["full_free"]):
            def_surf = ['symp1','symp2','wall1','wall2','wall3','far','outflow']
        else:
            def_surf = ['symp1','symp2','wall1','wall2','wall3','inflow','far','outflow']
        struct_surf = 'wall2'

        aero_builder = ADflowBuilder(options=aero_options,  scenario="aerodynamic", def_surf=def_surf, struct_surf=struct_surf) #mesh_options=warp_options,
        aero_builder.initialize(self.comm)
        # aero_builder.solver.addFunction('cd','wall2','cd_def')
        # aero_builder.solver.addFunction('cdv','wall2','cdv_def')
        # aero_builder.solver.addFunction('cdp','wall2','cdp_def')
        # aero_builder.solver.addFunction('cdm','wall2','cdm_def')
        aero_builder.solver.addFunction('drag','allWalls','d_def')
        # aero_builder.solver.addFunction('drag',None,'d_def')
        aero_builder.solver.addFunction('dragviscous','allWalls', 'dv_def')
        # aero_builder.solver.addFunction('dragviscous',None, 'dv_def')
        aero_builder.solver.addFunction('dragpressure','allWalls','dp_def')
        # aero_builder.solver.addFunction('dragpressure',None,'dp_def')
        aero_builder.solver.addFunction('dragmomentum','allWalls','dm_def')
        # aero_builder.solver.addFunction('dragmomentum',None,'dm_def')
        # aero_builder.solver.addFunction('cd',aero_builder.solver.allWallsGroup,'cd_def')
        # aero_builder.solver.addFunction('cdv',aero_builder.solver.allWallsGroup,'cdv_def')
        # aero_builder.solver.addFunction('cdp',aero_builder.solver.allWallsGroup,'cdp_def')
        # aero_builder.solver.addFunction('cdm',aero_builder.solver.allWallsGroup,'cdm_def')

        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # # component to determine post-shock flow properties for ADFlow
        if(self.options["use_shock_comp"]):
            dvs.add_output("shock_angle", opt_options["shock_angle"])
            self.add_subsystem("shock", ShockAngleComp())

        else:
            dvs.add_output("M1", val=impinge_setup.mach)
            dvs.add_output("beta", val=impinge_setup.beta)
            dvs.add_output("P1", val=impinge_setup.P)
            dvs.add_output("T1", val=impinge_setup.T)
            # dvs.add_output("r1", val=impinge_setup.rho)

        if(self.options["use_inflow_comp"]):
            dvs.add_output("M0", 3.0)
            dvs.add_output('P0', 2919.0)
            dvs.add_output('T0', 217.0)
            self.add_subsystem("upstream", InflowComp())
        else:
            dvs.add_output("vx0", impinge_setup.VX)
            dvs.add_output('P0', impinge_setup.P0)
            dvs.add_output('r0', impinge_setup.r0)



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
            P = impinge_setup.P, 
            T = impinge_setup.T, 
            # rho = impinge_setup.rho, 
            # evalFuncs=["cd_def", "cdv_def","cdm_def","cdp_def"]
            evalFuncs=["d_def", "dv_def","dm_def","dp_def"]
            # evalFuncs=["drag", "dragviscous","dragpressure","dragmomentum"]
        )

        # modify to allow for additional BCDVS
        # should be safe
        BCVarFuncs = ["Pressure", "PressureStagnation", "Temperature", "Density", "DensityStagnation", "VelocityX", "TemperatureStagnation", "Thrust", "Heat"]
        ap.possibleBCDVs = set(BCVarFuncs)


        ap.addDV("mach", value=impinge_setup.mach, name="mach1")
        ap.addDV("beta", value=impinge_setup.beta, name="beta")
        ap.addDV("P", value=impinge_setup.P, name="pressure1")
        ap.addDV("T", value=impinge_setup.T, name="temperature1")
        # ap.addDV("rho", value=impinge_setup.rho, name="density1")

        if(not self.options["full_free"]):
            if self.options["subsonic"]:
                ap.setBCVar("PressureStagnation", impinge_setup.P0, "inflow")
                ap.setBCVar("DensityStagnation",  impinge_setup.r0, "inflow")

                ap.addDV("PressureStagnation", family="inflow", name="pressure0")
                ap.addDV("DensityStagnation",  family="inflow", name="density0")
            else:
                ap.setBCVar("Pressure", impinge_setup.P0, "inflow")
                ap.setBCVar("Density",  impinge_setup.r0, "inflow")
                ap.setBCVar("VelocityX", impinge_setup.VX, "inflow")

                ap.addDV("Pressure", family="inflow", name="pressure0")
                ap.addDV("Density",  family="inflow", name="density0")
                ap.addDV("VelocityX", family="inflow", name="velocityx0")

        self.test.coupling.mphys_set_ap(ap)
        self.test.aero_post.mphys_set_ap(ap)

        if(self.options["use_shock_comp"]):
            self.connect("shock_angle", "shock.shock_angle")
            self.connect("M0", "shock.mach0")
            self.connect("P0", "shock.P0")
            self.connect("T0", "shock.T0")

            self.connect("shock.mach1", "test.coupling.mach1")
            self.connect("shock.flow_angle", "test.coupling.beta")
            self.connect("shock.T1", "test.coupling.temperature1")
            self.connect("shock.P1", "test.coupling.pressure1")
            self.connect("shock.mach1", "test.aero_post.mach1")
            self.connect("shock.flow_angle", "test.aero_post.beta")
            self.connect("shock.T1", "test.aero_post.temperature1")
            self.connect("shock.P1", "test.aero_post.pressure1")
        else:
            self.connect("M1", "test.coupling.mach1")
            self.connect("M1", "test.aero_post.mach1")
            self.connect("beta", "test.coupling.beta")
            self.connect("beta", "test.aero_post.beta")
            self.connect("P1", "test.coupling.pressure1")
            self.connect("P1", "test.aero_post.pressure1")
            self.connect("T1", "test.coupling.temperature1")
            self.connect("T1", "test.aero_post.temperature1")
            # self.connect("r1", "test.coupling.density1")
            # self.connect("r1", "test.aero_post.density1")
        
        if(not self.options["full_free"]):
            if(self.options["use_inflow_comp"]):
                self.connect("M0", "upstream.mach0")
                self.connect("P0", "upstream.P0")
                self.connect("T0", "upstream.T0")

                if not self.options["subsonic"]:
                    self.connect("upstream.VelocityX", "test.coupling.velocityx0")
                self.connect("upstream.Density", "test.coupling.density0")
                self.connect("upstream.Pressure", "test.coupling.pressure0")
                if not self.options["subsonic"]:
                    self.connect("upstream.VelocityX", "test.aero_post.velocityx0")
                self.connect("upstream.Density", "test.aero_post.density0")
                self.connect("upstream.Pressure", "test.aero_post.pressure0")
            else:
                if not self.options["subsonic"]:
                    self.connect("vx0", "test.coupling.velocityx0")
                self.connect("r0", "test.coupling.density0")
                self.connect("P0", "test.coupling.pressure0")
                if not self.options["subsonic"]:
                    self.connect("vx0", "test.aero_post.velocityx0")
                self.connect("r0", "test.aero_post.density0")
                self.connect("P0", "test.aero_post.pressure0")


if __name__ == '__main__':

    ################################################################################
    # OpenMDAO setup
    ################################################################################

    use_shock = False
    use_inflow = False
    full_far = False
    subsonic = False

    problem_settings = default_impinge_setup
    # problem_settings.aeroOptions['equationType'] = 'laminar NS'
    problem_settings.aeroOptions['equationType'] = 'Euler'
    # problem_settings.aeroOptions['NKSwitchTol'] = 1e-6 #1e-6
    problem_settings.aeroOptions['NKSwitchTol'] = 1e-3 #1e-6
    problem_settings.aeroOptions['nCycles'] = 5000000

    problem_settings.aeroOptions['L2Convergence'] = 1e-15
    problem_settings.aeroOptions['printIterations'] = False
    problem_settings.aeroOptions['printTiming'] = True

    problem_settings.mach = 1.7
    # problem_settings.beta = 0.8
    # problem_settings.mach = 0.8

    if full_far:
        aeroGridFile = f'../meshes/imp_TEST_73_73_25.cgns'
    elif subsonic:
        aeroGridFile = f'../meshes/imp_subs_73_73_25.cgns'
    else:
        aeroGridFile = f'../meshes/imp_mphys_73_73_25.cgns'
    problem_settings.aeroOptions['gridFile'] = aeroGridFile


    prob = om.Problem()
    prob.model = Top(problem_settings=problem_settings, subsonic=subsonic,
                                                         use_shock_comp=use_shock, 
                                                         use_inflow_comp=use_inflow, 
                                                         full_free=full_far)

    # prob.model.add_design_var("M1")
    # prob.model.add_design_var("beta")
    prob.model.add_design_var("P1")
    prob.model.add_design_var("T1")
    # prob.model.add_design_var("r1")

    # prob.model.add_design_var("P0")
    # prob.model.add_design_var("M0")
    # prob.model.add_design_var("T0")
    # prob.model.add_design_var("vx0")
    # prob.model.add_design_var("r0")
    # prob.model.add_design_var("P0")

    # prob.model.add_design_var("shock_angle")

    # prob.model.add_objective("test.aero_post.cd_def")
    # prob.model.add_objective("test.aero_post.cdv_def")
    # prob.model.add_objective("test.aero_post.cdp_def")
    # prob.model.add_objective("test.aero_post.cdm_def")
    # prob.model.add_objective("test.aero_post.d_def")
    # prob.model.add_objective("test.aero_post.dv_def")
    prob.model.add_objective("test.aero_post.dp_def")
    # prob.model.add_objective("test.aero_post.dm_def")
    # prob.model.add_objective("test.aero_post.drag")
    # prob.model.add_objective("test.aero_post.dragviscous")
    # prob.model.add_objective("test.aero_post.dragpressure")
    # prob.model.add_objective("test.aero_post.dragmomentum")

    prob.setup(mode='rev')
    om.n2(prob, show_browser=False, outfile="mphys_as_adflow_eb_%s_2pt.html")
    # prob.set_val("mach", default_impinge_setup.mach)
    # prob.set_val("beta", 10.0)
    
    prob.run_model()
    shock_comp_outs = [
        "shock.mach1",
        "shock.flow_angle",
        "shock.T1", 
        "shock.P1",
    ]
    shock_comp_ins = [
        "shock.mach0",
        "shock.shock_angle",
        "shock.P0", 
        "shock.T0",
    ]
    # prob.check_partials(includes=shock_comp_ins + shock_comp_outs)
    funcDerivs = {}
    prob.model.test.coupling.solver.solver.evalFunctionsSens(prob.model.test.coupling.solver.ap, funcDerivs)

    prob.check_totals(step_calc='rel_avg')
    import pdb; pdb.set_trace()

    #prob.model.list_outputs()

    if MPI.COMM_WORLD.rank == 0:
        print("cd = %.15f" % prob["test.aero_post.cd_def"])
    #     prob.model.


"""

i'm thinking rgas might be the culprit
nah


pref
timeref (prob not)
href
uref
winf
rgas

muref (maybe in viscous case?)


IT HAS TO DO WITH THE INTEGRATION SURFACES FOR SURE, PREFD GETS THE RIGHT VALUE
AT SOME POINT IF FAMILY GROUP IS SET TO NONE (ALL SURFACES)

-----------------------
Group: Top 'Full Model'
-----------------------
  Full Model: 'test.aero_post.dp_def' wrt 'dvs.P1'
    Analytic Magnitude: 4.548233e-01
          Fd Magnitude: 7.712190e-01 (fd:None)
    Absolute Error (Jan - Jfd) : 3.163957e-01 *

    Relative Error (Jan - Jfd) / Jfd : 4.102541e-01 *

    Raw Analytic Derivative (Jfor)
[[0.45482328]]

    Raw FD Derivative (Jfd)
[[0.77121903]]


ARE WE IN MASTER_B
 MASTER_B 1
 MASTER_B 2
 MASTER_B 3
 MASTER_B 4
 MADE IT TO APPLYALLBC_BLOCK_B
 MADE OUT OF APPLYALLBC_BLOCK_B
 MASTER_B 5
 DID WE MAKE IT?
 prefd before:   0.48397762634064073     
 prefd after :   0.77186737058245858     
 tempd1   (muref):    0.0000000000000000     
           murefd:    0.0000000000000000     
 tempd2    (rgas):    0.0000000000000000     
            rgasd:    0.0000000000000000     
 tempd3 (timeref):    0.0000000000000000     
         timerefd:    0.0000000000000000     
            prefd:   0.77186737058245858     
Solving linear in mphys_adflow
Solving ADjoint Transpose with PETSc...
Solving ADjoint Transpose with PETSc time (s) =     0.00
 Norm of error = 0.2228E-05    Iterations =  106
 ------------------------------------------------
 PETSc solver converged after   106 iterations.
 ------------------------------------------------
 ARE WE IN MASTER_B
 MASTER_B 1
 MASTER_B 2
 MASTER_B 3
 MASTER_B 4
 MADE IT TO APPLYALLBC_BLOCK_B
 MADE OUT OF APPLYALLBC_BLOCK_B
 MASTER_B 5
 DID WE MAKE IT?
 prefd before:    0.0000000000000000     
 prefd after :  -0.31704409046229221     
 tempd1   (muref):    0.0000000000000000     
           murefd:    0.0000000000000000     
 tempd2    (rgas):    0.0000000000000000     
            rgasd:    0.0000000000000000     
 tempd3 (timeref):    0.0000000000000000     
         timerefd:    0.0000000000000000     
            prefd:  -0.31704409046229221     


    print *, "tempd1   (muref): ", tempd1
    print *, "          murefd: ", murefd
    print *, "tempd2    (rgas): ", tempd2
    print *, "           rgasd: ", rgasd
    print *, "tempd3 (timeref): ", tempd3
    print *, "        timerefd: ", timerefd
    print *, "           prefd: ", prefd
    print *, "           pinfd: ", pinfd

CLOSER WITH UREF ACCOUNTED
STILL NOT QUITE THERE



WHAT'S MISSING?


pinfd is cancelling itself out at some point, don't think it's relevant

are we not taking rhoref into account (P T mach definition)

Divide rhoinfdimd by 72,917.89627702

DRHO/DT = -P/RT^2 = -2.6927513082979661213487587983942e-4

default mach
AD = -0.15494944
rhoinfdimd = 651.30941027172992 
DRHO/DP*DDRAG/DRHO = 0.008932092716
adding this makes AD -0.14601734728304, FD = -0.1454
T FD = -0.12589514
DRHO/DT*DDRAG/DRHO = -0.17538142666159775162793

mach 4.0
DRHO/DP*DDRAG/DRHO = 0.008932092716
adding this makes AD -0.169298663387, FD = -0.1685429

mach 1.7
AD = -0.08414637
rhoinfdimd = -1758.4695206470465
DRHO/DP*DDRAG/DRHO = 0.008932092716
adding this makes AD -0.1082621167566878552, FD = -0.10748369
T FD = 0.53660872
DRHO/DT*DDRAG/DRHO = 0.473512110232443181178

IT SEEMS CONSISTENT, BUT IT LOOKS LIKE THERE'S A TINY ERROR LEFT


"""