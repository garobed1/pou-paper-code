import scipy.sparse as sps
import numpy as np
from integration import CalcElemStiff, CalcElemLoad

def StiffAssemble(L, E, Iyy, Nelem):
    """
    Assemble the global sparse stiffness matrix
    
    Inputs:
        L - length of the beam
        E - longitudinal elastic modulus
        Iyy - moment of inertia with respect to the y axis, as function of x
        Nelem - number of finite elements to use
    Outputs:
        A - sparse global stiffness matrix
    """

    A = np.zeros(2*Nelem,2*Nelem)

    dx = L/Nelem
    for i in range(Nelem):
        Aelem = CalcElemStiff(E, Iyy[i], Iyy[i+1], dx)
        A[(i-2)*2+1:i*2, (i-2)*2+1:i*2] += Aelem