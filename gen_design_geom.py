import numpy
import openmdao.api as om
import plate_ffd as pf
import math
from mpi4py import MPI
from idwarp import USMesh
from baseclasses import *
from adflow import ADFLOW
from pygeo import DVGeometry, DVConstraints
from plate_comp_opts import aeroOptions, warpOptions, optOptions

'''Just generate a mesh moved with a set of design variables'''

aoptions = aeroOptions
woptions = warpOptions
ooptions = optOptions

# Generate FFD and DVs
DVGeo = pf.createFFD()

# starting flat mesh
meshname = aoptions['gridFile']

mesh = USMesh(options=warpOptions)
coords = mesh.getSurfaceCoordinates()

DVGeo.addPointSet(coords, "coords")

vars = optOptions['ro_shape']
dvDict = {'pnts':vars}
DVGeo.setDesignVars(dvDict)
new_coords = DVGeo.update("coords")

# move the mesh using idwarp
mesh.setSurfaceCoordinates(new_coords)
mesh.warpMesh()

# Write the new grid file.
mesh.writeGrid(f'opt_result.cgns')
        


