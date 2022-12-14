import numpy as np
from beam.utils import Error

"""
Evaluates Cubic Hermitian Basis Functions and their first and second derivatives on the interval [-1,1]

4 nodes total, 4 Basis functions evaluated at each point xi
"""

def cubicHermite(xi, dx):
    """
    Computes the values of cubic Hermitian basis functions on the interval [-1,1], for a given element length

    Inputs:
        xi - point at which to evaluate the shape functions
        dx - element length
    Outputs:
        N - 4x1 vector containing the shape functions at xi
    """ 

    # if (dx <= 0.0):
    #     raise Error("element length must be strictly positive")
    # if (xi < -1.0) | (xi > 1.0):
    #     raise Error("shape functions must be evaluated in the interval [-1,1]")

    # return 4 values, one for each node 

    N = np.zeros(4)

    workp = (1 + xi)**2
    workm = (1 - xi)**2

    N[0] = 0.25*workm*(2 + xi)
    N[1] = 0.125*dx*workm*(1 + xi)
    N[2] = 0.25*workp*(2 - xi)
    N[3] = -0.125*dx*workp*(1 - xi)

    return N

def cubicHermiteD(xi, dx):
    """
    Computes the first derivatives of cubic Hermitian basis functions on the interval [-1,1], for a given element length
    
    Inputs:
        xi - point at which to evaluate the shape functions
        dx - element length
    Outputs:
        dN - 4x1 vector containing the shape function derivatives at xi
    """ 

    if (dx <= 0.0):
        raise Error("element length must be strictly positive")
    if (xi < -1.0) | (xi > 1.0):
        raise Error("shape functions must be evaluated in the interval [-1,1]")

    # return 4 values, one for each node 

    dN = np.zeros(4)

    workp = (1 + xi)**2
    workm = (1 - xi)**2

    dN[0] = -0.5*(1 - xi)*(2 + xi) + 0.25*workm
    dN[1] = -0.25*dx*(1 - xi)*(1 + xi) + 0.125*dx*workm
    dN[2] = 0.5*(1 + xi)*(2 - xi) - 0.25*workp
    dN[3] = -0.25*dx*(1 + xi)*(1 - xi) + 0.125*dx*workp

    return dN/dx

def cubicHermiteD2(xi, dx):
    """
    Computes the second derivatives of cubic Hermitian basis functions on the interval [-1,1], for a given element length
    
    Inputs:
        xi - point at which to evaluate the shape functions
        dx - element length
    Outputs:
        dN - 4x1 vector containing the shape function second derivatives at xi
    """ 

    if (dx <= 0.0):
        raise Error("element length must be strictly positive")
    if (xi < -1.0) | (xi > 1.0):
        raise Error("shape functions must be evaluated in the interval [-1,1]")

    # return 4 values, one for each node 

    ddN = np.zeros(4)

    workp = (1 + xi)**2
    workm = (1 - xi)**2

    ddN[0] = 6*xi/dx
    ddN[1] = 3*xi - 1
    ddN[2] = -6*xi/dx
    ddN[3] = 3*xi + 1

    return ddN/dx

