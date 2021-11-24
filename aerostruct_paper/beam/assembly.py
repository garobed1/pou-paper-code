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

    A = np.zeros([2*(Nelem-1),2*(Nelem-1)])

    dx = L/Nelem
    # interior loop
    for i in range(2,Nelem):
        Aelem = CalcElemStiff(E, Iyy[i-1], Iyy[i], dx)
        A[(i-2)*2:i*2, (i-2)*2:i*2] += Aelem
    

    # boundary in here for now, fixed at left end
    Aelem = CalcElemStiff(E, Iyy[0], Iyy[1], dx)
    A[0:2, 0:2] += Aelem[2:4, 2:4]

    # fix at both ends? 
    Aelem = CalcElemStiff(E, Iyy[Nelem-1], Iyy[Nelem], dx)
    A[(2*(Nelem-1))-2:(2*(Nelem-1)), (2*(Nelem-1))-2:(2*(Nelem-1))] += Aelem[0:2, 0:2]

    Asp = sps.csr_matrix(A)
    return Asp

def LoadAssemble(L, f, Nelem):
    """
    Assemble the global load vector
    
    Inputs:
        L - length of the beam
        f - longitudinal elastic modulus
        Nelem - number of finite elements to use
    Outputs:
        b - global load vector
    """

    b = np.zeros(2*(Nelem-1))

    dx = L/Nelem
    # interior loop
    for i in range(2,Nelem):
        belem = CalcElemLoad(f[i-1], f[i], dx)
        b[(i-2)*2:i*2] += belem

    # boundary in here for now, fixed at left end
    belem = CalcElemLoad(f[0], f[1], dx)
    b[0:1] += belem[2:3]

    # fix at both ends? 
    belem = CalcElemLoad(f[Nelem-1], f[Nelem], dx)
    b[(2*(Nelem-1))-2:(2*(Nelem-1))] += belem[0:2]

    return b