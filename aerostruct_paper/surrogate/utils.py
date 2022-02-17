import numpy as np
from scipy.linalg import lstsq





def quadraticSolve(x, xn, f, fn, g, gn):

    """
    Construct a quadratic interpolation over a limited neighborhood of points 
    about a given point, for which the function values and gradients are known.
    
    solve : fh_i(x_j) = f_j = f_i + g_i(x_j-x_i) + (x_j-x_i)^T H_i (x_j-x_i) 
    and   : gh_i(x_j) = g_j = g_i + H_i(x_j-x_i)
    unknown : {f_i, g_i, H_i} or if we use data, {H_i}

    in a least squares sense.

    Inputs: 
        x - point to center the approximation
        xn - neighborhood of points to attempt interpolation through
        f - function value at center point
        fn - function values of neighborhood points
        g - gradient at center point
        gn - gradient at neighborhood points

    Outputs
        fc - solved center function value
        gc - solved center gradient
        Hc - solved center Hessian
    """
    N = g.shape[0] # problem dimension
    M = 1 + xn.shape[0] # number of points to fit
    vN = sum(range(N+1))
    csize = 1 + N + vN # number of unknowns,  f, g, H
    rsize = M + M*N    # number of equations, fh, gh conditions 

    # solution vector, stored as {fc, gc1, gc2, .., gN, H }
    sol = np.zeros(1 + N + vN)

    mat = np.zeros([rsize, csize])
    rhs = np.zeros(rsize)

    # assemble rhs
    rhs[0:M] = np.append(f, fn)
    for i in range(M):
        if(i == 0):
            gvec = g
        else:
            gvec = gn[i-1,:]

        rhs[(M + i*N):(M + i*N + N)] = gvec
    import pdb; pdb.set_trace()

    





def quadratic(x, x0, f0, g, h):
    """
    Given the gradient and Hessian about a nearby point, return the quadratic
    Taylor series approximation of the function
    
    f(x) = f(x0) + g(x0)^T*(x-x0) + (1/2)*(x-x0)^T*h(x0)*(x-x0) + O((x-x0)^3)

    Inputs:
        x - point to evaluate the approximation
        x0 - center point of the Taylor series
        f0 - function value at the center
        g - gradient at the center
        h - Hessian at the center
    Outputs:
        f - quadratic Taylor series approximation at x
    """

    dx = x - x0

    Hdx = np.matmul(h,dx.T)
    dHd = np.dot(dx,Hdx)
    f = f0 + np.dot(g,dx) + 0.5*dHd

    return f


def neighborhood(i, trx):
    """
    Determine an "optimal" neighborhood around a data point for estimating the 
    Hessian, based on the closest points that best surround the point
    
    Inputs:
        i - index of point to determine neighborhood of
        trx - full list of data points
    Outputs:
        ind - indices of points to include in the neighborhood
    """
    ind = []
    return ind


def symMatfromVec(i, j, N):
    """
    Retrieve the index to query a symmetric matrix stored in a compressed vector form

    Taken from https://stackoverflow.com/questions/3187957/how-to-store-a-symmetric-matrix

    Inputs:
        i - row index
        j - column index
        N - matrix size
    Outputs:
        k - 1d symmetric matrix index

    matrix: 0 1 2 3
              4 5 6
                7 8
                  9

    vector: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    """
    if(i <= j):
        return i*N - (i - 1) * i/2 + j - i
    else:
        return j*N - (j - 1) * j/2 + i - j
    

x = np.array([1, 2, 3, 4])
xn = np.zeros([6, 4])

f = 10
fn = np.zeros(6)
for i in range(6):
    fn[i] = i

g = x
gn = xn
for i in range(6):
    gn[i,:] = i

quadraticSolve(x, xn, f, fn, g, gn)