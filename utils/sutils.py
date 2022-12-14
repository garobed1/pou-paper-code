import numpy as np
from numpy.linalg import qr
from scipy.linalg import lstsq, lu_factor, lu_solve, solve, inv, eig
from scipy.spatial.distance import cdist
from functions.example_problems import Heaviside, MultiDimJump, Quad2D
from smt.problems import RobotArm
from smt.sampling_methods import LHS

def standardization2(X, y, bounds):

    """

    Scale X and y, X according to its bounds, and y according to its mean and standard 
    deviation

    Parameters
    ----------

    X: np.ndarray [n_obs, dim]
            - The input variables.

    y: np.ndarray [n_obs, 1]
            - The output variable.

    bounds: np.ndarray [dim, 2]

    Returns
    -------

    X: np.ndarray [n_obs, dim]
          The standardized input matrix.

    y: np.ndarray [n_obs, 1]
          The standardized output vector.

    X_offset: list(dim)
            The mean (or the min if scale_X_to_unit=True) of each input variable.

    y_mean: list(1)
            The mean of the output variable.

    X_scale:  list(dim)
            The standard deviation (or the difference between the max and the
            min if scale_X_to_unit=True) of each input variable.

    y_std:  list(1)
            The standard deviation of the output variable.

    """

    X_offset = bounds[:,0]
    X_max = bounds[:,1]
    X_scale = X_max - X_offset

    X_scale[X_scale == 0.0] = 1.0
    y_mean = np.mean(y, axis=0)
    y_std = y.std(axis=0, ddof=1)
    y_std[y_std == 0.0] = 1.0

    # scale X and y
    X = (X - X_offset) / X_scale
    y = (y - y_mean) / y_std
    return X, y, X_offset, y_mean, X_scale, y_std

def innerMatrixProduct(A, x):
    """
    Return scalar result of x^T A x
    """
    return np.dot(np.dot(x.T, A), x)

def getDirectCovariance(r, dr, d2r, theta, nt, ij):
    """
    Construct each block of the direct-gradient covariance matrix for direct gradient methods

    Takes the block form [P Pg.T; Pg S]

    Inputs: 
        r - symmetric covariances
        dr - symmetric covariance derivatives
        d2r - symmetric covariance second derivatives
        theta - hyperparameters
        nt - number of training points
        ij - symmetric indexing
    Outputs:
        P - Covariance matrix
        Pg - Covariance Jacobian
        S - Covariance Hessian
    """
    n_elem = r.shape[0]
    nc = theta.shape[0]
    
    # P
    P = np.eye(nt)
    P[ij[:, 0], ij[:, 1]] = r[:, 0]
    P[ij[:, 1], ij[:, 0]] = r[:, 0]

    # Pg
    Pg = np.zeros([nt*nc, nt])
    for k in range(n_elem):
        Pg[(ij[k,1]*nc):(ij[k,1]*nc+nc), ij[k,0]] = dr[k].T
        Pg[(ij[k,0]*nc):(ij[k,0]*nc+nc), ij[k,1]] = -dr[k].T

    # S
    S = np.zeros([nt*nc, nt*nc])#*(2*theta)
    for k in range(nt):
        S[k*nc:(k+1)*nc, k*nc:(k+1)*nc] = np.diag(2*theta)

    for k in range(n_elem):
        S[(ij[k,0]*nc):(ij[k,0]*nc+nc), (ij[k,1]*nc):(ij[k,1]*nc+nc)] = d2r[k]
        S[(ij[k,1]*nc):(ij[k,1]*nc+nc), (ij[k,0]*nc):(ij[k,0]*nc+nc)] = d2r[k]

    return P, Pg, S

def quadraticSolve(x, xn, f, fn, g, gn):

    """
    Construct a quadratic interpolation over a limited neighborhood of points 
    about a given point, for which the function values and gradients are known.
    
    solve : fh_i(x_j) = f_j = f_i + g_i(x_j-x_i) + (x_j-x_i)^T H_i (x_j-x_i) 
    and   : gh_i(x_j) = g_j = g_i + H_i(x_j-x_i)
    unknown : {f_i, g_i, H_i} or if we use data, {H_i}

    in a minimum norm least squares sense.

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
    sol = np.zeros(csize)

    mat = np.zeros([rsize, csize])
    rhs = np.zeros(rsize)
    dx = xn
    for i in range(M-1):
        dx[i,:] -= x

    # assemble rhs
    rhs[0:M] = np.append(f, fn)
    for i in range(M):
        if(i == 0):
            gvec = g
        else:
            gvec = gn[i-1,:]

        rhs[(M + i*N):(M + i*N + N)] = gvec

    # assemble mat

    # function fitting
    mat[0:M,0] = 1
    for j in range(N):
        mat[1:M, j+1] = dx[:,j]

    for k in range(1,M):
        for i in range(N):
            for j in range(N):
                ind = symMatfromVec(i,j,N)
                mat[k, 1+N+ind] += 0.5*dx[k-1,i]*dx[k-1,j]

    # gradient fitting
    for j in range(N):
        mat[M+j::N, j+1] = 1

    for k in range(1,M):
        for i in range(N):
            for j in range(N):
                ind = symMatfromVec(i,j,N)
                mat[M+k*N+i, 1+N+ind] += dx[k-1,j]

    # now solve the system in a least squares sense
    # rhs[0] *= 100
    # mat[0,0] *= 100
    #import pdb; pdb.set_trace()
    sol = lstsq(mat, rhs)
    # LU, PIV = lu_factor(mat)
    #sol = solve(mat, rhs)

    #import pdb; pdb.set_trace()

    fh = sol[0][0]
    gh = sol[0][1:N+1]
    Hh = sol[0][N+1:N+1+vN]

    #import pdb; pdb.set_trace()
    return fh, gh, Hh


def quadraticSolveHOnly(x, xn, f, fn, g, gn):

    """
    Construct a quadratic interpolation over a limited neighborhood of points 
    about a given point, for which the function values and gradients are known.
    
    solve : fh_i(x_j) = f_j = f_i + g_i(x_j-x_i) + (x_j-x_i)^T H_i (x_j-x_i) 
    and   : gh_i(x_j) = g_j = g_i + H_i(x_j-x_i)
    unknown : {f_i, g_i, H_i} or if we use data, {H_i}

    in a minimum norm least squares sense.

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
    M = xn.shape[0] # number of points to fit
    vN = sum(range(N+1))
    csize = vN # number of unknowns, H
    rsize = M+M*N    # number of equations, gh conditions 

    # solution vector, stored as { H }
    sol = np.zeros(csize)

    mat = np.zeros([rsize, csize])
    rhs = np.zeros(rsize)
    dx = xn
    for i in range(M):
        dx[i,:] -= x

    # assemble rhs
    
    for i in range(M):
        rhs[i] = fn[i] - f - np.dot(g, dx[i,:]) 
        gvec = gn[i,:] - g
        rhs[(M+i*N):(M+i*N + N)] = gvec

    # assemble mat
    # function fitting
    for k in range(0,M):
        for i in range(N):
            for j in range(N):
                ind = symMatfromVec(i,j,N)
                mat[k, ind] += 0.5*dx[k,i]*dx[k,j]

    # gradient fitting
    for k in range(M):
        for i in range(N):
            for j in range(N):
                ind = symMatfromVec(i,j,N)
                mat[M+k*N+i, ind] += dx[k,j]

    # now solve the system in a least squares sense
    # rhs[0] *= 100
    # mat[0,0] *= 100
    #import pdb; pdb.set_trace()
    sol = lstsq(mat, rhs)
    # LU, PIV = lu_factor(mat)
    #sol = solve(mat, rhs)

    #import pdb; pdb.set_trace()

    Hh = sol[0]

    #import pdb; pdb.set_trace()
    return Hh




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

def linear(x, x0, f0, g):
    """
    Given the gradient and Hessian about a nearby point, return the linear
    Taylor series approximation of the function
    
    f(x) = f(x0) + g(x0)^T*(x-x0) + O((x-x0)^2)

    Inputs:
        x - point to evaluate the approximation
        x0 - center point of the Taylor series
        f0 - function value at the center
        g - gradient at the center

    Outputs:
        f - linear Taylor series approximation at x
    """

    dx = x - x0

    f = f0 + np.dot(g,dx)

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
        return int(i*N - (i - 1) * i/2 + j - i)
    else:
        return int(j*N - (j - 1) * j/2 + i - j)


def maxEigenEstimate(x, xn, g, gn):

    """
    Find the maximum eigenvalue pair of the sample-space projected Hessian at x, 
    in the neighborhood xn. 

    HX \approx G, H - Hessian, G - Gradient
    
    X = xn.T - x, QR = X, Q.T H Q \approx Q.T G R

    Find eigenpairs of 0.5(Q.T G R^-1 + R^-1.T G.T Q), \lambda, vhat

    Then the corresponding eigenvector of H is v = Qvhat

    Inputs: 
        x - point to center the approximation
        xn - neighborhood of points to attempt interpolation through
        g - gradient at center point
        gn - gradient at neighborhood points

    Outputs
        evalm - max eigenvalue
        evecm - corresponding eigenvector
    """
    N = g.shape[0] # problem dimension
    M = xn.shape[0] # number of points to fit

    # get matrix of neighborhood distances
    dx = xn
    for i in range(M):
        dx[i,:] -= x
    dx = dx.T

    # get gradient matrix
    dg = gn
    for i in range(M):
        dg[i,:] -= g
    G = dg.T

    # QR factorize the dx matrix
    Q, R = qr(dx, mode='reduced')

    # Invert upper part of R
    Rinv = inv(R)

    # Create the approximation of the projected Hessian
    M = np.matmul(Q.T, np.matmul(G, Rinv))

    # Find the eigenvalues of the symmetric part
    evals, evecs = eig(0.5*(M + M.T))

    # Find the largest eigenpair and estimate the Hessian eigenvector
    o = np.argsort(abs(evals))
    evalm = evals[o[-1]]
    evecmtilde = evecs[:,o[-1]]
    evecm = np.matmul(Q, evecmtilde)
    return evalm, evecm


# return intersection points of line with the problem bounds through exhaustive search

# this assumes we know the line intersects with the box, and our origin is inside
#TODO: Write a test for this
def boxIntersect(xc, xdir, bounds):

    m = xc.shape[0]
    large = 1e100
    blims = np.zeros(m*2)

    for j in range(m):
        if(abs(xdir[j]) < 1e-12):
            blims[j] = np.sign(bounds[j,0] - xc[j])*large
            blims[m+j] = np.sign(bounds[j,1] - xc[j])*large
        else:
            blims[j] = (bounds[j,0] - xc[j])/xdir[j] # all element-wise
            blims[m+j] = (bounds[j,1] - xc[j])/xdir[j]

    # find minimum positive alpha
    p1 = min([i for i in blims if i >= 0])

    # find maximum negative alpha
    p0 = max([i for i in blims if i <= 0])

    # edge cases
    if(p0 == p1):
        p1b = min([i for i in blims if i > 0])
        p0b = max([i for i in blims if i < 0])
        x1b = xc + p1b*xdir
        x0b = xc + p0b*xdir

        # check which point is in bounds
        if all((x1b - bounds[:,0]) >= 0) and all(x1b - bounds[:,1] <= 0):
            p1 = p1b
        elif all((x0b - bounds[:,0]) >= 0) and all(x0b - bounds[:,1] <= 0):
            p0 = p0b
        else: #corner case, hope this never happens
            p0 = 0.0
            p1 = 0.0

    return p0, p1

def divide_cases(ncases, nprocs):
    """
    From parallel OpenMDAO beam problem example

    Divide up adaptive sampling runs among available procs.

    Parameters
    ----------
    ncases : int
        Number of load cases.
    nprocs : int
        Number of processors.

    Returns
    -------
    list of list of int
        Integer case numbers for each proc.
    """
    data = []
    for j in range(nprocs):
        data.append([])

    wrap = 0
    for j in range(ncases):
        idx = j - wrap
        if idx >= nprocs:
            idx = 0
            wrap = j

        data[idx].append(j)

    return data


def estimate_pou_volume(trx, bounds):
    """
    Estimate volume of partition-of-unity basis cells by counting closest randomly-distributed points

    Parameters
    ----------
    trx: np.ndarray
        list of training point locations

    bounds: np.ndarray
        bounds of the domain, expected to be unit hypercube

    Returns
    -------
    dV: np.ndarray
        list of cell volumes ordered like trx
    """
    m, n = trx.shape

    dV = np.ones(m)
    sampling = LHS(xlimits=bounds)
    vx = sampling(100*m)
    ms, ns = vx.shape
    dists = cdist(vx, trx)

    # loop over the random points
    for i in range(ms):
        cind = np.argmin(dists[i])
        dV[cind] += 1.

    dV /= ms

    return dV

# dim = 2
# trueFunc = MultiDimJump(ndim=dim)
# bounds = trueFunc.xlimits
# boxIntersect(np.array([-2.5, 0.5]), np.array([-.707,-.707]), bounds)

# dim = 2
# trueFunc = RobotArm(ndim=dim)
# xlimits = trueFunc.xlimits

# xc = np.array([0.9, np.pi])
# xdir = np.array([1, 4])
# xdir = xdir/np.linalg.norm(xdir)

# p0, p1 = boxIntersect(xc, xdir, xlimits)
# import pdb; pdb.set_trace()


# dim = 2
# trueFunc = Quad2D(ndim=dim, theta=np.pi/4)
# xlimits = trueFunc.xlimits
# sampling = LHS(xlimits=xlimits)

# nt0  = 3

# t0 = np.array([[0.25, 0.75],[0.8, 0.5],[0.75, 0.1]])# sampling(nt0)[0.5, 0.5],
# f0 = trueFunc(t0)
# g0 = np.zeros([nt0,dim])
# for i in range(dim):
#     g0[:,i:i+1] = trueFunc(t0,i)

# quadraticSolveHOnly(t0[0,:], t0[1:3,:], f0[0], f0[1:3], g0[0,:], g0[1:3,:])





# x = np.array([1, 2, 3, 4])
# xn = np.zeros([6, 4])
# for i in range(6):
#     xn[i,:] = 0.5*i

# f = 10
# fn = np.zeros(6)
# for i in range(6):
#     fn[i] = i

# g = x
# gn = xn
# for i in range(6):
#     gn[i,:] = i

# quadraticSolve(x, xn, f, fn, g, gn)