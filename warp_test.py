from mpi4py import MPI
from idwarp import USMesh
from numpy import cos, arctan as atan

options = {
  'gridFile':'grid_struct_273x193_vol_mod2.cgns',
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

# Create the mesh object
mesh = USMesh(options=options, comm=MPI.COMM_WORLD)

# Extract all coordinates
coords0 = mesh.getSurfaceCoordinates()

new_coords = coords0.copy()

#form bump arc according to parameters
l = 1.0
l0 = 0.5 
b = 0.05

l1 = l+l0
lc = (l/2.0) + l0
r = ((l/2.0)**2 + b**2)/(2.0*b)
rmb = r - b

for i in range(len(coords0)):
     x = new_coords[i, 0]
     y = new_coords[i, 1]
     if x > l0 and x < l1:
         th = atan(abs(x - lc)/rmb)
         new_coords[i, 2] += r*cos(th) - rmb
#    new_coords[i, :] *= 1.1


# Reset the newly computed surface coordiantes
mesh.setSurfaceCoordinates(new_coords)

# Actually run the mesh warping
mesh.warpMesh()

# Write the new grid file.
mesh.writeGrid('warped.cgns')