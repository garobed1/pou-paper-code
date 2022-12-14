
from example_problems import ToyLinearScale, Ishigami, Peaks2D, QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgSingleHump, FuhgP3, FuhgP8, FuhgP9, FuhgP10, FakeShock
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, WingWeight
from shock_problem import ImpingingShock
from mpi4py import MPI

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
    elif(prob == "shock"):
        xlimits = np.zeros([dim,2])
        xlimits[0,:] = [23., 27.]
        xlimits[1,:] = [0.36, 0.51]
        trueFunc = ImpingingShock(ndim=dim, input_bounds=xlimits, comm=MPI.COMM_SELF)
    else:
        raise ValueError("Given problem not valid.")

    return trueFunc
