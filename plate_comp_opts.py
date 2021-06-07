gridFile = f'test_mfmc_121_25_1.cgns'
probName = 'pure_mc_2000'

astar = [0.28788225, 0.29621159, 0.34267779, 0.18972925, 0.19873263, 0.30842999,
    0.27963309, 0.23999633, 0.28788225, 0.29621159, 0.34267779, 0.18972925,
    0.19873263, 0.30842999, 0.27963309, 0.23999633]

dist = [
    [0.42015014, 0.131375,   0.6375,     0.75      ],
    [0.48949273, 0.135875,   0.6875,     0.95      ],
    [0.38547884, 0.129125,   0.6125,     0.65      ],
    [0.45482143, 0.133625,   0.6625,     0.85      ]]

gridFilesML = [
f'test_mfmc_41_25_1.cgns',
f'test_mfmc_81_25_1.cgns',
f'test_mfmc_121_25_1.cgns'
]

optOptions = { #general optimization parameters
    'prob_name':probName,
    'NX':20, #number of x FFD points, not necessarily the number of design vars
    'bumpBounds':[1.00, 2.00], #ends of the bump
    'mach':0.85, #inflow mach number
    'Re':3000000, #inflow reynolds number
    'DVFraction':0.1, #fraction of NX on either side of bump control points not used as DVs
    'DVUpperBound':2.0,  #upper bound for control point movement
    'DVLowerBound':0.0,  #lower bound for control point movement (set to 0 when thickness constraints work)
    'DVInit':0.3,  #uniform initial design state
    'DCMinThick':0.01,  #uniform minimum thickness
    'DCMinArea':1.0025,  #minimum surface area, if used
    'DCThickFrac':0.75, #percentage of bump area to constrain
    'constrain_opt':True,
    'use_area_con':True,
    'check_partials':False,  #check partial derivatives
    'run_once':True, # run a single iteration of the model with given settings
    'ro_shape':astar # shape variables 
}

uqOptions = { #general UQ parameters
    'mode':'MC', # MC: Normal Monte Carlo with LHS points
                 # SC: Stochastic Collocation
                 # MFMC: Multi-Fidelity Monte Carlo with LHS points
                 # MLMC: Multi-Level Monte Carlo with LHS points
    'MCTimeBudget':False, #determine number of samples by time budget option 'P'
    'MCPure':True, #don't use LHS points
    'SCPts':3, # number of sc points, 2*SCPts - 1 order of SC polynomial
    'NS':28, #number of sample points
    'NS0':8, #start up sample number for multi-level
    'rho':2., #robust objective std dev ratio
    'use-predetermined-samples':False, #input N1 at each level instead of running MLMC
    'predet-N1':[3,4,5], #user-determined N1
    'predet-a1':[1,1.1,1.2], #user-determined a1 for MFMC
    'dist':dist, #distribution to use ONLY IF running the model once
    'gridFileLevels':gridFilesML, #all available meshes for multi-level (need at least 3)
    'vartol': 2e-5, #ML variance tolerance for convergence
    'P':600. #computational budget in seconds
}

aeroOptions = { #ADflow aero solver options
    # Common Parameters
    'gridFile':gridFile,
    'outputDirectory':'./results/',
    'writeTecplotSurfaceSolution':False,
    'writeSurfaceSolution':False,
    'writeVolumeSolution':False,
    
    # Physics Parameters
    'equationType':'RANS',
    'turbulenceModel':'SA',
    'turbulenceProduction':'vorticity',
    'useft2SA':True,
    'eddyVisInfRatio':3.0,
    # [kappa, cb1,    cb2,   sigma,         cv1, cw2, cw3, ct1, ct2, ct3, ct4, rot]
    # [0.41,  0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0]
    'SAConsts':[0.41, 0.1355, 0.622, 0.66666666667, 7.1, 0.3, 2.0, 1.0, 2.0, 1.2, 0.5, 2.0],
    
    # Common Parameters
    'MGCycle':'sg',
    'nCycles':100000,
    'monitorvariables':['resrho','resmom','cd','resturb'],
    'useNKSolver':True,
    'NKSwitchTol':1e-6,
    'NKSubspaceSize':200,
    'NKLS':'none',
    'useANKSolver':True,
    'ANKCoupledSwitchTol':1e-5,
    'ANKConstCFLStep':0.4,
    'ANKCFLLimit':1000000000.0,
    'L2Convergence':1e-12,
    
    # Design options
    'meshSurfaceFamily':'allSurfaces',
    'designSurfaceFamily':'allSurfaces',

    # Adjoint options
    'adjointL2Convergence': 1e-06,

    # Output
    'volumeVariables':['eddyratio','mach'],
    'surfaceVariables':['yplus'],
    'printIterations':False,
    'printTiming':False,
    'printWarnings':False,
    'setMonitor':False
    }

warpOptions = { #IDwarp mesh movement options
  'gridFile':gridFile,
  'fileType':'CGNS',
  'specifiedSurfaces':None,
  'symmetrySurfaces':None,
  'symmetryPlanes':[[[0.0, 0.0, 0.0],[0.0, -1.0, 0.0]],[[0.0, 1.0, 0.0],[0.0, 1.0, 0.0]]],
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

