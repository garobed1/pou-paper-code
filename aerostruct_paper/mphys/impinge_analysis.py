import imp
import numpy as np
import argparse
from mpi4py import MPI
import sys
sys.path.insert(1,"../beam/")

import openmdao.api as om

from mphys import Multipoint

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
            isym = 0
            ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym)
            ldxfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS Setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        dvs.add_output("dv_struct", np.array(ndv_struct * [0.05]))

        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
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

        self.connect("dv_struct", f"{scenario}.dv_struct")

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

        self.test.coupling.aero.mphys_set_ap(ap)
        self.test.aero_post.mphys_set_ap(ap)


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
prob.setup()
#om.n2(prob, show_browser=False, outfile="mphys_as_adflow_eb_%s_2pt.html")
#prob.run_model()
prob.check_partials()

#prob.model.list_outputs()

# if MPI.COMM_WORLD.rank == 0:
#     #print("cd = ", prob["test.aero_post.cd_def"])
#     prob.model.