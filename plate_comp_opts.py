optOptions = { #general optimization parameters
    'prob_name':'constrained_grid_c_2_10',
    'NX':10, #number of x FFD points, not necessarily the number of design vars
    'bumpBounds':[1.35, 1.55], #ends of the bump
    'DVFraction':0.1, #fraction of NX on either side of bump control points not used as DVs
    'DVUpperBound':0.3,  #upper bound for control point movement
    'DVLowerBound':0.08,  #lower bound for control point movement (set to 0 when thickness constraints work)
    'DVInit':0.1,  #uniform initial design state
    'DCMinThick':0.01,  #uniform minimum thickness
    'DCThickFrac':0.75, #percentage of bump area to constrain
    'constrain_opt':True,
    'check_partials':False  #check partial derivatives
}

aeroOptions = { #ADflow aero solver options
    # Common Parameters
    'gridFile':f'grid_c.cgns',
    'outputDirectory':'./results/',
    'writeTecplotSurfaceSolution':True,
    
    # Physics Parameters
    'equationType':'RANS',
    'turbulenceModel':'SA',
    'turbulenceProduction':'vorticity',
    'useft2SA':True,
    'eddyVisInfRatio':3.0,
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
    'L2Convergence':1e-13,
    
    # Design options
    'meshSurfaceFamily':'allSurfaces',
    'designSurfaceFamily':'allSurfaces',

    # Output
    'volumeVariables':['eddy','eddyratio','dist'],
    'printIterations':False,
    'printTiming':False,
    'printWarnings':False
    }

warpOptions = { #IDwarp mesh movement options
  'gridFile':'grid_c.cgns',
  'fileType':'CGNS',
  'specifiedSurfaces':None,
  'symmetrySurfaces':None,
  'symmetryPlanes':[],
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