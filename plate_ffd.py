import numpy as np
from pygeo import DVGeometry, geo_utils
from idwarp import USMesh
from plate_comp_opts import warpOptions, optOptions

# Create the FFD file and DVGeometry object
def createFFD():

    # Bounding box for bump
    x_root_range = [-0.1, 3.1]
    y_root_range = [-0.1, 1.1]
    z_root_range = [-0.1, 1.1]

    # Number of FFD control points per dimension
    nX = 10  # streamwise
    nY = 2  
    nZ = 2   # only modify the top control points

    # Compute grid points
    span_dist = np.linspace(0, 1, nX)
    x_sections = span_dist * (x_root_range[1] - x_root_range[0]) + x_root_range[0]
    z_sections = z_root_range

    X = np.zeros((nX, nY, nZ))
    Y = np.zeros((nX, nY, nZ))
    Z = np.zeros((nX, nY, nZ))
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                X[i,j,k] = x_sections[i]
                Y[i,j,k] = y_root_range[j]
                Z[i,j,k] = z_sections[k]
    # rst Write
    # Write FFD to file
    filename = "plateffdp.xyz"
    f = open(filename, "w")
    f.write("\t\t1\n")
    f.write("\t\t%d\t\t%d\t\t%d\n" % (nX, nY, nZ))
    for set in [X, Y, Z]:
        for k in range(nZ):
            for j in range(nY):
                for i in range(nX):
                    f.write("\t%3.8f" % (set[i,j,k]))
                f.write("\n")
    f.close()

    # Create the child FFD, actual DVs defined on this

    # Bounding box for bump
    x_root_range_c = optOptions['bumpBounds']
    y_root_range_c = [-0.05, 1.05]
    z_root_range_c = [-0.05, 0.95]

    # Number of FFD control points per dimension
    nXc = optOptions['NX'] # streamwise
    nYc = 2  
    nZc = 3   # only modify the top control points

    # Compute grid points
    span_dist_c = np.linspace(0, 1, nXc)
    x_sections_c = span_dist_c * (x_root_range_c[1] - x_root_range_c[0]) + x_root_range_c[0]
    z_span_dist_c = np.linspace(0, 1, nZc)
    z_sections_c = z_span_dist_c * (z_root_range_c[1] - z_root_range_c[0]) + z_root_range_c[0]

    Xc = np.zeros((nXc, nYc, nZc))
    Yc = np.zeros((nXc, nYc, nZc))
    Zc = np.zeros((nXc, nYc, nZc))
    for i in range(nXc):
        for j in range(nYc):
            for k in range(nZc):
                Xc[i,j,k] = x_sections_c[i]
                Yc[i,j,k] = y_root_range_c[j]
                Zc[i,j,k] = z_sections_c[k]
    # rst Write
    # Write FFD to file
    filenamec = "plateffdc.xyz"
    fc = open(filenamec, "w")
    fc.write("\t\t1\n")
    fc.write("\t\t%d\t\t%d\t\t%d\n" % (nXc, nYc, nZc))
    for set in [Xc, Yc, Zc]:
        for k in range(nZc):
            for j in range(nYc):
                for i in range(nXc):
                    fc.write("\t%3.8f" % (set[i,j,k]))
                fc.write("\n")
    fc.close()

    # define the design variables
    
    #add point set here? no
    # meshOptions = warpOptions
    # mesh = USMesh(options=meshOptions)
    # coords = mesh.getSurfaceCoordinates()

    DVGeo = DVGeometry(filename)
    
    #DVGeo.addPointSet(coords, "coords")
    DVGeoc = DVGeometry(filenamec, child=True)
    DVGeoc.addRefAxis('dummy_axis', xFraction=0.1, alignIndex='i')
    DVGeo.addChild(DVGeoc)

    # local design vars are just the Z-positions of (some) upper control points
    length = x_root_range_c[1] - x_root_range_c[0]
    z_mid = (z_root_range_c[1] + z_root_range_c[0])/2
    frac = optOptions['DVFraction']
    P1 = [x_root_range_c[0]+frac*length, 0, z_mid/2]
    P2 = [x_root_range_c[1]-frac*length, 0, 3*z_mid/2]
    PS = geo_utils.PointSelect(psType = 'y', pt1=P1, pt2=P2)
    

    #
    vname="pnts"
    UB = optOptions['DVUpperBound']
    DVGeoc.addGeoDVLocal(dvName=vname, lower=0.0, upper=UB, axis="z", scale=1, pointSelect=PS)

    return DVGeo



# # test out on a mesh
# gridFile = "grid_struct_69x49_vol_mod2.cgns"
# meshOptions = warpOptions
# meshOptions["gridfile"] = gridFile
# mesh = USMesh(options=meshOptions)
# coords = mesh.getSurfaceCoordinates()

# DVGeo.addPointSet(coords, "coords")
# dvDict = DVGeoc.getValues()
# #for i in range(int(nXc/2)):
# dvDict["pnts"][:] = 0.2
# # dvDict["pnts"][2] = 0.2
# # dvDict["pnts"][3] = 0.2
# # dvDict["pnts"][4] = 0.1
# # dvDict["pnts"][5] = 0.1
# # dvDict["pnts"][6] = 0.1
# # dvDict["pnts"][7] = 0.1
# DVGeoc.setDesignVars(dvDict)
# DVGeoc.printDesignVariables()
# new_coords = DVGeo.update("coords")
# DVGeoc.writePlot3d("ffdc_deformed.xyz")
# #DVGeo.writePointSet("coords", "surf")

# # move the mesh using idwarp

# # Reset the newly computed surface coordiantes
# mesh.setSurfaceCoordinates(new_coords)

# # Actually run the mesh warping
# mesh.warpMesh()

# # Write the new grid file.
# mesh.writeGrid(f'ffd_warped.cgns')