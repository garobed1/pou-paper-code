from scipy.sparse.linalg.dsolve import spsolve
import numpy as np
from scipy.sparse.linalg.dsolve.linsolve import spsolve_triangular
from hermite_basis import cubicHermite, cubicHermiteD, cubicHermiteD2
import assembly as asm
from utils import Error

"""
Implementing the solver as a class with the ability to call the solver, a setup function, 
and easily set force vectors, beam shape parameters, and get output functions as needed

May write a base class if it seems appropriate
"""

class EulerBeamSolver():
    """
    Solves the Euler-Bernoulli beam equation to estimate displacements for a beam fixed at both ends
    Uses a finite-element method with cubic Hermitian basis functions. The beam is assumed to lie along 
    the x axis, with the force applied transversely in the xz plane.

    Can be initialized without settings, but needs to be set in order to call the solver. Settings are 
    given in a dictionary object

    Settings:
        name - solver name

        L - beam length

        E - longitudinal elastic modulus

        Nelem - number of elements to use

        Iyy - y axis moment of inertia as a function of x, size (Nelem+1)

        force - force per unit length along the beam axis x, size (Nelem+1)
    """

    def __init__(self, settings=None):

        # declare global variables and flags
        # flags
        self.req_update = True
        self.req_setup = True
        self.req_solve = True

        # variables
        self.name = None
        self.Nelem = 0
        self.L = 0
        self.E = 0
        self.force = None
        self.Iyy = None

        # stiffness matrix and load vector
        self.A = None
        self.b = None

        # solution vector
        self.u = None
        self.res = None

        # call setup if we have settings
        if(settings):
            self.setup(settings)

    def __call__(self):

        if(self.req_setup):
            Error("Must call setup(settings) at least once before attemping to solve")
        if(self.req_update):
            self.assemble()

        # solve
        self.u = spsolve(self.A, self.b)

        self.req_solve = False
        return self.u

    def getSolution(self):

        if(self.req_solve):
            self.__call__()

        return self.u

    def getResidual(self):

        if(self.req_setup):
            Error("Must call setup(settings) at least once before attemping to solve")

        if(self.req_update):
            self.assemble()

        # multiply through
        self.res = self.A*self.u - self.b

        return self.res

    def setup(self, settings):

        self.name = settings["name"]
        self.Nelem = settings["Nelem"]
        self.L = settings["L"]
        self.E = settings["E"]
        self.force = settings["force"]
        self.Iyy = settings["Iyy"]
        self.th = settings["th"]
        
        #set left bound of beam in x just for mesh transfer purposes
        if "l_bound" in settings:
            self.bounds = [settings["l_bound"], settings["l_bound"] + self.L ] 
        else:
            self.bounds = [0.0, self.L]

        self.u = np.zeros(2*(self.Nelem-1))
        self.res = np.zeros(2*(self.Nelem-1))

        if(self.Iyy == None):
            if "th" in settings:
                self.computeRectMoment(settings["th"])
            else:
                Error("If Iyy not supplied directly, we must compute it from th")

        # assemble matrix and load vector
        self.assemble()

        self.req_setup = False

    def assemble(self):

        self.A = asm.StiffAssemble(self.L, self.E, self.Iyy, self.Nelem)
        self.b = asm.LoadAssemble(self.L, self.force, self.Nelem)

        # need to find a way to apply BCs
        # set flag when the system is assembled
        self.req_update = False


    # shortcuts to quickly set some settings
    def setLoad(self, force):

        self.force = force
        self.req_update = True
        self.req_solve = True

    def setIyy(self, Iyy):

        self.Iyy = Iyy
        self.req_update = True
        self.req_solve = True

    def setThickness(self, th):

        self.th = th
        self.computeRectMoment(self.th)
        self.req_update = True
        self.req_solve = True

    def computeRectMoment(self, th):

        # Compute moment of an (infinitely) rectangular beam, given thickness

        # technically moment per unit length, since Iyy = w*h^3/12, and width goes "into page"
        Iyy = np.zeros(self.Nelem + 1, dtype='complex_')

        for i in range(self.Nelem + 1):
            Iyy[i] = th[i]*th[i]*th[i]/12.

        self.setIyy(Iyy)

    def getMeshPoints(self):

        # Return mesh node locations in space

        pts = np.zeros(self.Nelem+1)
        dx = self.L/self.Nelem
        for i in range(pts.size):
            pts[i] = self.bounds[0] + i*dx

        return pts

    def evalFunctions(self, func_list):

        # compute all functions

        # element-wise stress, don't compute if we only want mass
        #if(not all((func_list, ["mass"]))):
        dx = self.L/self.Nelem 
        sigma = np.zeros(self.Nelem+1, dtype='complex_')
        zero = np.array([0])
        utrue = np.concatenate((zero,zero,self.u,zero,zero))
        for i in range(self.Nelem):
            xi = [-1,1]
            d2Nl = cubicHermiteD2(xi[0], dx)
            d2Nr = cubicHermiteD2(xi[1], dx)
            sigma[i] = 0.5*self.E*self.th[i]*np.dot(d2Nl,utrue[i*2:(i+1)*2+2])
            sigma[i+1] = 0.5*self.E*self.th[i+1]*np.dot(d2Nr,utrue[i*2:(i+1)*2+2])

        sigma = sum(sigma)
        # mass

        mass = self.evalMass()
        
        dict = {}

        for key in func_list:
            if(key == "mass"):
                dict["mass"] = mass
            if(key == "stress"):
                dict["stress"] = sigma

        return dict

    def evalMass(self):

        mass = 0
        dx = self.L/self.Nelem 
        for i in range(self.Nelem):
            mass += 0.5*(self.th[i]+self.th[i+1])*dx

        return mass

    def evalthSens(self, func):

        # complex step
        h = 1e-10

        gdict = {} 

        for key in func:
            gdict[key] = np.zeros(len(self.th))
        
        thc = np.zeros(len(self.th), dtype='complex_')
        for i in range(len(self.th)):
            thc.real = self.th
            thc.imag = np.zeros(len(self.th))
            thc[i] = thc[i] + h*1j
            self.setThickness(thc)
            self.__call__()
            sol = self.evalFunctions(func)

            for key in func:
                gdict[key][i] = np.imag(sol[key])/h
        
        # reset
        self.setThickness(self.th)
        return gdict

    def evalforceSens(self, func):

        # complex step
        h = 1e-10

        gdict = {} 

        for key in func:
            gdict[key] = np.zeros(len(self.force))
        
        fc = np.zeros(len(self.force), dtype='complex_')
        for i in range(len(self.force)):
            fc.real = self.force
            fc.imag = np.zeros(len(self.force))
            fc[i] = fc[i] + h*1j
            self.setLoad(fc)
            self.__call__()
            sol = self.evalFunctions(func)

            for key in func:
                gdict[key][i] = np.imag(sol[key])/h

        # reset
        self.setLoad(self.force)
        return gdict

# Nelem = 20

# settings = {
#     # "name":"hello",
#     # "Nelem":10,
#     # "L":4,
#     # "E":300,
#     # "force":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     # "Iyy":None,
#     # "th":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     "name":"hello",
#     "Nelem":Nelem,
#     "L":0.254, #0.254, 
#     "E":400000,
#     "force":np.ones(Nelem+1)*1.0,
#     "Iyy":None,
#     "th":np.ones(Nelem+1)*0.01,
#     "l_bound":2.0,
# }

# beamsolve = EulerBeamSolver(settings)


# func_list = ["mass","stress"]

# csdict = beamsolve.evalthSens(func_list)

# h = 1e-7

# #finite difference
# fddict = {} 

# th = settings["th"]
# beamsolve.setThickness(th)
# beamsolve()
# dict = beamsolve.evalFunctions(func_list)
# s0 = dict["stress"]
# fd = np.zeros(len(settings["th"]))
# for i in range(len(settings["th"])):
#     thc = np.array(th)
#     thc[i] = thc[i] + h
#     beamsolve.setThickness(thc)
#     beamsolve()
#     dict = beamsolve.evalFunctions(func_list)
#     fd[i] = (dict["stress"]-s0)/h

# import pdb; pdb.set_trace()

