import numpy as np

from smt.problems.problem import Problem

class ALOSDim(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[1, 2, 3], types=int)
        self.options.declare("name", "ALOSDim", types=str)

    def _setup(self):
        self.xlimits[:, 0] = 0.0
        self.xlimits[:, 1] = 1.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        dim = self.options["ndim"]
        a = 30. if dim == 1 else 21.
        b = 0.9 if dim == 1 else 0.7

        y = np.zeros((ne, 1), complex)
        if kx is None:
            y[:,0] = np.sin(a*(x[:,0] - 0.9)**4)*np.cos(2.*(x[:,0] - 0.9)) + (x[:,0] - b)/2.
            if dim > 1:
                y[:,0] += 2.*x[:,1]*x[:,1]*np.sin(x[:,0]*x[:,1])
            if dim > 2:
                y[:,0] += 3.*x[:,2]*x[:,2]*np.sin(x[:,0]*x[:,1]*x[:,2])
        elif kx == 0:
            y[:,0] = np.cos(a*(x[:,0] - 0.9)**4)*np.cos(2.*(x[:,0] - 0.9))*(4*a*(x[:,0] - 0.9)**3)
            y[:,0] += -np.sin(a*(x[:,0] - 0.9)**4)*np.sin(2.*(x[:,0] - 0.9))*2.
            y[:,0] += 0.5
            if dim > 1:
                y[:,0] += 2.*x[:,1]*x[:,1]*x[:,1]*np.cos(x[:,0]*x[:,1])
            if dim > 2:
                y[:,0] += 3.*x[:,1]*x[:,2]*x[:,2]*x[:,2]*np.cos(x[:,0]*x[:,1]*x[:,2])
        elif kx == 1:
            y[:,0] = 0.
            if dim > 1:
                y[:,0] += 4.*x[:,1]*np.sin(x[:,0]*x[:,1]) + 2.*x[:,1]*x[:,1]*x[:,0]*np.cos(x[:,0]*x[:,1])
            if dim > 2:
                y[:,0] += 3.*x[:,0]*x[:,2]*x[:,2]*x[:,2]*np.cos(x[:,0]*x[:,1]*x[:,2])
            
        elif kx == 2:
            y[:,0] = 0.
            if dim > 2:
                y[:,0] += 6.*x[:,2]*np.sin(x[:,0]*x[:,1]*x[:,2]) +  3.*x[:,0]*x[:,1]*x[:,2]*x[:,2]*np.cos(x[:,0]*x[:,1]*x[:,2])

        return y

class ScalingExpSine(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, types=int)
        self.options.declare("name", "ScalingExpSine", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -2.0
        self.xlimits[:, 1] = 2.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        dim = self.options["ndim"]
        pi = np.pi
        
        y = np.zeros((ne, 1), complex)
        if kx is None:
            for i in range(dim):
                y[:,0] += np.exp(-0.1*x[:,i])*np.sin(0.5*pi*x[:,i])
        
        elif kx is not None:
            y[:,0] = -0.1*np.exp(-0.1*x[:,kx])*np.sin(0.5*pi*x[:,kx])
            y[:,0] += 0.5*pi*np.exp(-0.1*x[:,kx])*np.cos(0.5*pi*x[:,kx])
            
        y /= dim
        return y
    
if __name__ == '__main__':

    x_init = 0.
    # N = [5, 3, 2]
    # xlimits = np.array([[-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.]])
    # pdfs =  [x_init, 'uniform', 'uniform', 'uniform']
    # samp1 = CollocationSampler(np.array([x_init]), N=N,
    #                             xlimits=xlimits, 
    #                             probability_functions=pdfs, 
    #                             retain_uncertain_points=True)
    
    import matplotlib.pyplot as plt
    from smt.problems import Rosenbrock
    from utils.sutils import convert_to_smt_grads

    ndir = 150

    # 1d ALOS
    func1 = ALOSDim(ndim=1)
    xlimits1 = func1.xlimits
    x1 = np.linspace(xlimits1[0][0], xlimits1[0][1], ndir)
    x1 = np.atleast_2d(x1).T
    TF1 = func1(x1)

    # Plot the target function
    plt.plot(x1, TF1, "-k", label=f'True')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    plt.title(r"ALOS Function 1D")
    #plt.legend(loc=1)
    plt.savefig(f"./1dALOS.pdf", bbox_inches="tight")
    plt.clf()


    # 2d ALOS
    func2 = ALOSDim(ndim=2)
    xlimits2 = func2.xlimits
    x1 = np.linspace(xlimits2[0][0], xlimits2[0][1], ndir)
    x2 = np.linspace(xlimits2[1][0], xlimits2[1][1], ndir)
    X1, X2 = np.meshgrid(x1, x2)
    combx = np.concatenate((X1.reshape((ndir*ndir, 1)), X2.reshape((ndir*ndir, 1))), axis=1)
    TF2 = func2(combx)
    TF2 = TF2.reshape((ndir, ndir))

    # Plot the target function
    plt.contourf(X1, X2, TF2, levels=30, label=f'True')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"ALOS Function 2D")
    plt.colorbar()
    #plt.legend(loc=1)
    plt.savefig(f"./2dALOS.pdf", bbox_inches="tight")
    plt.clf()

    # 1d ExpSin
    func1 = ScalingExpSine(ndim=1)
    xlimits1 = func1.xlimits*10
    x1 = np.linspace(xlimits1[0][0], xlimits1[0][1], ndir)
    x1 = np.atleast_2d(x1).T
    TF1 = func1(x1)

    # Plot the target function
    plt.plot(x1, TF1, "-k", label=f'True')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    plt.title(r"ExpSine Function 1D")
    #plt.legend(loc=1)
    plt.savefig(f"./1dES.pdf", bbox_inches="tight")
    plt.clf()


    # 2d ExpSin
    func2 = ScalingExpSine(ndim=2)
    xlimits2 = func2.xlimits*10
    x1 = np.linspace(xlimits2[0][0], xlimits2[0][1], ndir)
    x2 = np.linspace(xlimits2[1][0], xlimits2[1][1], ndir)
    X1, X2 = np.meshgrid(x1, x2)
    combx = np.concatenate((X1.reshape((ndir*ndir, 1)), X2.reshape((ndir*ndir, 1))), axis=1)
    TF2 = func2(combx)
    TF2 = TF2.reshape((ndir, ndir))

    # Plot the target function
    plt.contourf(X1, X2, TF2, levels=30, label=f'True')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"ExpSine Function 2D")
    plt.colorbar()
    #plt.legend(loc=1)
    plt.savefig(f"./2dES.pdf", bbox_inches="tight")
    plt.clf()

    import pdb; pdb.set_trace()