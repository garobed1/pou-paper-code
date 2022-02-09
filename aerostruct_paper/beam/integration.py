import numpy as np
from utils import Error
from math import sqrt
from hermite_basis import cubicHermite, cubicHermiteD, cubicHermiteD2

def CalcElemStiff(E, IL, IR, dx):
    """
    Computes the element stiffness matrix for the Euler-Bernoulli equation

    Inputs:
        E - longitudinal elastic modulus
        IL - moment of inertia at left side of element
        IR - moment of inertia at right side of element
        dx - length of the element
    Outputs:
        Aelem - the 4x4 element stiffness matrix
    """ 
    if (IL <= 0.0) | (IR <= 0.0) | (E <= 0.0) | (dx <= 0.0):
        Error('Inputs must all be strictly positive')

    Aelem = np.zeros([4,4], dtype='complex_')

    # get quadrature points
    xi, w = GaussQuad(2)

    qsize = len(xi)
    # assemble element stiffness one quadrature point at a time
    for i in range(qsize):
        B = cubicHermiteD2(xi[i], dx)
        MI = (IL*(1-xi[i]) + IR*(1+xi[i]))*0.5
        Int = (0.5*E*dx)*MI*np.outer(B,B)
        Aelem = Aelem + (w[i]*Int)

    return Aelem


def CalcElemLoad(qL, qR, dx):
    """
    Computes the element load vector for the Euler-Bernoulli equation

    Inputs:
        qL - force per unit length at left side of element
        qR - force per unit length at right side of element
        dx - length of the element
    Outputs:
        belem - the 4x1 element load vector
    """ 
    if (dx <= 0.0):
        Error('Element length must be strictly positive')

    belem = np.zeros(4, dtype='complex_')

    # get quadrature points
    xi, w = GaussQuad(3)

    qsize = len(xi)

    # assemble element load one quadrature point at a time
    for i in range(qsize):
        N = cubicHermite(xi[i], dx)
        F = (qL*(1-xi[i]) + qR*(1+xi[i]))*0.5 # linear moment of inertia
        Int = 0.5*dx*F*N
        belem = belem + (w[i]*Int)

    return belem


def GaussQuad(n):
    """
    Returns gauss quadrature points and weights for use in integration, over the interval [-1,1]

    Inputs:
        n - quadrature order
    Outputs:
        xi - quadrature points
        w - quadrature weights
    """ 
    xi = np.zeros(n)
    w = np.zeros(n)

    if n == 1:
        xi[0] = 0.0
        w[0] = 2.0
    elif n == 2:
        xi[0] = -1/sqrt(3)
        w[0] = 1.0
        xi[1] = 1/sqrt(3)
        w[1] = 1.0
    elif n == 3:
        xi[0] = -sqrt(3/5)
        w[0] = 5/9
        xi[1] = 0.0
        w[1] = 8/9
        xi[2] = sqrt(3/5)
        w[2] = 5/9
    else:
        Error('GaussQuad is only implemented for n = 1, 2, or 3')

    return xi, w


# print(CalcElemStiff(10, 4, 3, 2))
# print(CalcElemLoad(4, 3, 2))