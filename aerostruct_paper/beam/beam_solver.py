from scipy.sparse.linalg.dsolve import spsolve
import numpy as np
from scipy.sparse.linalg.dsolve.linsolve import spsolve_triangular
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

        # call setup if we have settings
        if(settings):
            self.setup(settings)

    def __call__(self):

        if(self.req_setup):
            Error("Must call setup(settings) at least once before attemping to solve")

        if(self.req_update):
            self.assemble

        import pdb; pdb.set_trace()

        # solve
        self.u = spsolve(self.A, self.b)

        import pdb; pdb.set_trace()


    def setup(self, settings):

        self.name = settings["name"]
        self.Nelem = settings["Nelem"]
        self.L = settings["L"]
        self.E = settings["E"]
        self.force = settings["force"]
        self.Iyy = settings["Iyy"]

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

    def setIyy(self, Iyy):

        self.Iyy = Iyy
        self.req_update = True


settings = {
    "name":"hello",
    "Nelem":10,
    "L":4,
    "E":300,
    "force":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Iyy":[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
}

beamsolve = EulerBeamSolver(settings)

beamsolve()

