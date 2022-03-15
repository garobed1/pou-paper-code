# options file for the mphys test analysis

import numpy as np
#from tacs import TACS, elements, constitutive, functions

outputDirectory = './results'
aeroGridFile = f'../meshes/imp_mphys_73_73_25.cgns'
alpha = 0. #
beta = 7.2833969362749187
mach = 2.6381157549933598 #3.0
areaRef = 1.0
chordRef = 1.0
T = 254.02071103827234 #217.
P = 4987.6905797938707 #2919.
probName = 'impinge_mphys'

astar = [ 0.0,0.0]
dist = []
gridFilesML = []

optOptions = { #general optimization parameters
    'prob_name':probName,
    'NX':2, #number of x FFD points, not necessarily the number of design vars
    'bumpBounds':[1.00, 2.00], #ends of the bump
    'mach':0.1, #inflow mach number
    'Re':936000, #inflow reynolds number
    'DVFraction':0.1, #fraction of NX on either side of bump control points not used as DVs
    'DVUpperBound':2.0,  #upper bound for control point movement
    'DVLowerBound':0.0,  #lower bound for control point movement (set to 0 when thickness constraints work)
    'DVInit':0.0,  #uniform initial design state
    'DCMinThick':0.01,  #uniform minimum thickness
    'DCMinArea':1.0025,  #minimum surface area, if used
    'DCThickFrac':0.75, #percentage of bump area to constrain
    'constrain_opt':True,
    'use_area_con':True,
    'check_partials':False,  #check partial derivatives
    'run_once':True, # run a single iteration of the model with given settings
    'nRuns':1, # number of times to run the code, then print the average statistics for statistics runs
    'ro_shape':astar # shape variables 
}

uqOptions = { #general UQ parameters
    'mode':'SC', # MC: Normal Monte Carlo with LHS points
                 # SC: Stochastic Collocation
                 # MFMC: Multi-Fidelity Monte Carlo with LHS points
                 # MLMC: Multi-Level Monte Carlo with LHS points
    'MCTimeBudget':False, #determine number of samples by time budget option 'P'
    'MCPure':False, #don't use LHS points
    'SCPts':3, # number of SC points per direction, 2*SCPts - 1 order of SC polynomial
    'FullFactor': False, # if using SC, this tells it to do a full factorial analysis instead
    'ParamSlice':None, # if not none, take that param and compute slices, run once, and output the data
    'NS':40, #number of sample points
    'NS0':5, #start up sample number for multi-level
    'rho':1.0, #robust objective std dev ratio
    'use-predetermined-samples':False, #input N1 at each level instead of running MLMC
    'predet-N1':[3,4,5], #user-determined N1
    'predet-a1':[1,1.1,1.2], #user-determined a1 for MFMC
    'dist':dist, #distribution to use ONLY IF running the model once
    'gridFileLevels':gridFilesML, #all available meshes for multi-level (need at least 3)
    'vartol': 2e-5, #ML variance tolerance for convergence
    'P':1000. #computational budget in seconds
}

aeroOptions = { #ADflow aero solver options
    # Common Parameters
    'gridFile':aeroGridFile,
    'outputDirectory':'../results/',
    'writeTecplotSurfaceSolution':False,
    'writeSurfaceSolution':True,
    'writeVolumeSolution':True,
    
    # Physics Parameters
    'equationType':'RANS',
    'turbulenceModel':'SA',
    'turbulenceProduction':'vorticity',
    'useft2SA':True,
    'eddyVisInfRatio':3.0,
    # [kappa, cb1,    cb2,   sigma,         cv1, cw2, cw3, ct1, ct2, ct3, ct4, rot]
    # [0.41,  0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
    'SAConsts':[0.41,  0.1355, 0.6, 1.0, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0],
    
    # Common Parameters
    "CFL": 1.5,
    "CFLCoarse": 1.25,
    'MGCycle':'sg',
    'nCycles':100000,
    'monitorvariables':["resrho", "resturb"],
    'useNKSolver':True,
    'NKSwitchTol':1e-4,#e-1,
    'NKSubspaceSize':50,
    'NKPCILUFill':3,
    'NKLS':'none',
    'useANKSolver':True,
    'ANKCoupledSwitchTol':1e-3,
    'ANKConstCFLStep':0.4,
    'ANKCFLLimit':1000000000.0,
    "L2Convergence": 1e-12,
    # "L2ConvergenceCoarse": 1e-2,
    # "L2ConvergenceRel": 1e-4,
    "forcesAsTractions": False,
    
    # Design options
    'meshSurfaceFamily':'customSurface',
    #'designSurfaceFamily':'wall2',
    # Adjoint options
    'adjointL2Convergence': 1e-06,
    # Output
    'volumeVariables':['eddyratio','mach','cp','temp'],
    'surfaceVariables':['yplus','cf','cp','cfx','cfy','cfz'],
    'printIterations':True,
    'printTiming':True,
    'printWarnings':False,
    'setMonitor':False
    }

# Euler Bernoulli Structural Solver Options
nelem = 17
structOptions = {
    "name":probName,
    "Nelem":nelem,
    "L":0.254, #0.254, 
    "E":100000,
    "force":np.ones(nelem+1)*1.0,
    "Iyy":None,
    "th":np.ones(nelem+1)*0.001,
    "l_bound":2.0,
    "smax": 500,
    "get_funcs":["mass", "stresscon"]
    }

warpOptions = { #IDwarp mesh movement options
  'gridFile':aeroGridFile,
  'fileType':'CGNS',
  'specifiedSurfaces':None,
  'symmetrySurfaces':['sym1s','syms2'],
  'symmetryPlanes':None, # HOPE THIS WORKS, NO SYMMETRY PLANES
  #'symmetryPlanes':[[[0.0, 0.0, 0.0],[0.0, -1.0, 0.0]],[[0.0, 1.0, 0.0],[0.0, 1.0, 0.0]]],
  'aExp': 3.0,
  'bExp': 5.0,
  'LdefFact':1.0,
  'alpha':0.25,
  'errTol':0.0001,
  'evalMode':'fast',
  'useRotations':True,
  'zeroCornerRotations':True,
  'cornerAngle':30.0,
  'bucketSize':8,
}

