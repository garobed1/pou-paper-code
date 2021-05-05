gridFile = f'grid_foil_97_49_2.cgns'
probName = 'testmp7'

# astar = [
#     3.78162576e-01, 2.80605905e-01, 3.73461068e-01, 2.09207603e-01, 
#     2.81679376e-16, 4.45069938e-01, 2.13100484e-01, 3.44137984e-01, 
#     3.78162576e-01, 2.80605905e-01, 3.73461068e-01, 2.09207603e-01,
#     2.81679376e-16, 4.45069938e-01, 2.13100484e-01, 3.44137984e-01
#     ]

# dist = [
#     [0.42015014, 0.131375,   0.6375,     0.75      ],
#     [0.48949273, 0.135875,   0.6875,     0.95      ],
#     [0.38547884, 0.129125,   0.6125,     0.65      ],
#     [0.45482143, 0.133625,   0.6625,     0.85      ]]

astar = [
    3.84603790e-01, 2.63670133e-01, 3.77677847e-01, 2.08002805e-01,
    3.23468764e-16, 4.46386233e-01, 2.01538486e-01, 3.44862989e-01,
    3.84603790e-01, 2.63670133e-01, 3.77677847e-01, 2.08002805e-01,
    3.23468764e-16, 4.46386233e-01, 2.01538486e-01, 3.44862989e-01
    ]

dist = [
    [0.41148231, 0.1308125,  0.63125,    0.725     ],
    [0.37681102, 0.1285625,  0.60625,    0.625     ],
    [0.46348925, 0.1341875,  0.66875,    0.875     ],
    [0.44615361, 0.1330625,  0.65625,    0.825     ],
    [0.4808249,  0.1353125,  0.68125,    0.925     ],
    [0.42881796, 0.1319375,  0.64375,    0.775     ],
    [0.39414667, 0.1296875,  0.61875,    0.675     ],
    [0.49816055, 0.1364375,  0.69375,    0.975     ]]

astar_m75_20 =    [2.57421793e-01, 2.68722732e-01, 2.97207716e-01, 3.05921523e-01,
                    3.03159109e-01, 2.88756775e-01, 2.76741724e-01, 2.58832887e-01,
                    2.31751213e-01, 2.04331455e-01, 1.48283968e-01, 1.85935479e-01,
                    0.00000000e+00, 1.65986277e-01, 2.61946767e-18, 3.18325944e-02,
                    2.57421793e-01, 2.68722732e-01, 2.97207716e-01, 3.05921523e-01,
                    3.03159109e-01, 2.88756775e-01, 2.76741724e-01, 2.58832887e-01,
                    2.31751213e-01, 2.04331455e-01, 1.48283968e-01, 1.85935479e-01,
                    0.00000000e+00, 1.65986277e-01, 2.61946767e-18, 3.18325944e-02]

optOptions = { #general optimization parameters
    'prob_name':probName,
    'NX':10, #number of x FFD points, not necessarily the number of design vars
    'bumpBounds':[1.00, 2.00], #ends of the bump
    'mach':0.85, #inflow mach number
    'Re':3000000, #inflow reynolds number
    'DVFraction':0.1, #fraction of NX on either side of bump control points not used as DVs
    'DVUpperBound':2.0,  #upper bound for control point movement
    'DVLowerBound':0.0,  #lower bound for control point movement (set to 0 when thickness constraints work)
    'DVInit':0.1,  #uniform initial design state
    'DCMinThick':0.01,  #uniform minimum thickness
    'DCMinArea':1.005,  #minimum surface area, if used
    'DCThickFrac':0.75, #percentage of bump area to constrain
    'constrain_opt':True,
    'use_area_con':True,
    'check_partials':False,  #check partial derivatives
    'run_once':False, # run a single iteration of the model with given settings
    'ro_shape':astar_m75_20 # shape variables 
}

uqOptions = { #general UQ parameters
    'NS':4, #number of sample points
    'rho':2., #robust objective std dev ratio
    'dist':dist #distribution to use ONLY IF running the model once
}

aeroOptions = { #ADflow aero solver options
    # Common Parameters
    'gridFile':gridFile,
    'outputDirectory':'./results/',
    'writeTecplotSurfaceSolution':False,
    'writeSurfaceSolution':False,
    'writeVolumeSolution':True,
    
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
    'monitorvariables':['resrho','resmom','cl','cd','resturb'],
    'useNKSolver':True,
    'NKSwitchTol':1e-16,
    'NKSubspaceSize':200,
    'NKLS':'none',
    'useANKSolver':True,
    'ANKCoupledSwitchTol':1e-5,
    'ANKConstCFLStep':0.4,
    'ANKCFLLimit':1000000000.0,
    'L2Convergence':1e-08,
    
    # Design options
    'meshSurfaceFamily':'allSurfaces',
    'designSurfaceFamily':'allSurfaces',

    # Output
    'volumeVariables':['eddyratio'],
    'printIterations':False,
    'printTiming':True,
    'printWarnings':False
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

