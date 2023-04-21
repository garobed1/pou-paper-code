import numpy as np
from functions.example_problems import BetaRobust1D, BetaRobustEx1D, ToyLinearScale, Ishigami, Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgSingleHump, FuhgP3, FuhgP8, FuhgP9, FuhgP10, FakeShock
from functions.example_problems_2 import ALOSDim, ScalingExpSine, MixedSine
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, WingWeight
from functions.shock_problem import ImpingingShock
from mpi4py import MPI
import mphys_comp.impinge_setup as default_impinge_setup


def GetProblem(prob, dim, alpha = 8., use_design=False):
    # Problem Settings
    #alpha = 8.       #arctangent jump strength
    if(prob == "arctan"):
        trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
    elif(prob == "arctantaper"):
        trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
    elif(prob == "rosenbrock"):
        trueFunc = Rosenbrock(ndim=dim)
    elif(prob == "peaks"):
        trueFunc = Peaks2D(ndim=dim)
    elif(prob == "branin"):
        trueFunc = Branin(ndim=dim)
    elif(prob == "sphere"):
        trueFunc = Sphere(ndim=dim)
    elif(prob == "ishigami"):
        trueFunc = Ishigami(ndim=dim)
    elif(prob == "fuhgsh"):
        trueFunc = FuhgSingleHump(ndim=dim)
    elif(prob == "fuhgp3"):
        trueFunc = FuhgP3(ndim=dim)
    elif(prob == "fuhgp8"):
        trueFunc = FuhgP8(ndim=dim)
    elif(prob == "fuhgp9"):
        trueFunc = FuhgP9(ndim=dim)
    elif(prob == "fuhgp10"):
        trueFunc = FuhgP10(ndim=dim)
    elif(prob == "waterflow"):
        trueFunc = WaterFlow(ndim=dim)
    elif(prob == "weldedbeam"):
        trueFunc = WeldedBeam(ndim=dim)
    elif(prob == "robotarm"):
        trueFunc = RobotArm(ndim=dim)
    elif(prob == "cantilever"):
        trueFunc = CantileverBeam(ndim=dim)
    elif(prob == "hadamard"):
        trueFunc = QuadHadamard(ndim=dim)
    elif(prob == "toylinear"):
        trueFunc = ToyLinearScale(ndim=dim, use_design=use_design)
    elif(prob == "lpnorm"):
        trueFunc = LpNorm(ndim=dim)
    elif(prob == "wingweight"):
        trueFunc = WingWeight(ndim=dim)
    elif(prob == "fakeshock"):
        trueFunc = FakeShock(ndim=dim)
    elif(prob == "alos"):
        trueFunc = ALOSDim(ndim=dim)
    elif(prob == "expsine"):
        trueFunc = ScalingExpSine(ndim=dim)
    elif(prob == "mixedsine"):
        trueFunc = MixedSine(ndim=dim)
    elif(prob == "shock"):
        xlimits = np.zeros([dim,2])
        xlimits[0,:] = [23., 27.]
        xlimits[1,:] = [0.36, 0.51]

        problem_settings = default_impinge_setup
        problem_settings.aeroOptions['L2Convergence'] = 1e-15
        problem_settings.aeroOptions['printIterations'] = False
        problem_settings.aeroOptions['printTiming'] = False

        trueFunc = ImpingingShock(ndim=dim, input_bounds=xlimits, comm=MPI.COMM_SELF, problem_settings=problem_settings)
    elif(prob == "betatest"):
        trueFunc = BetaRobust1D(ndim=dim)
    elif(prob == "betatestex"):
        trueFunc = BetaRobustEx1D(ndim=dim)
    else:
        raise ValueError("Given problem not valid.")

    return trueFunc
