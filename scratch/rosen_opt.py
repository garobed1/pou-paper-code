import sys
import numpy as np
sys.path.insert(1,"../surrogate")
from problem_picker import GetProblem
from defaults import DefaultOptOptions
from optimizers import optimize
import matplotlib.pyplot as plt


prob = "rosenbrock"
dim = 2

func = GetProblem(prob, dim)

plt.rcParams['font.size'] = '18'
plt.rc('legend',fontsize=14)

noptions = DefaultOptOptions
noptions["gtol"] = 1e-5
noptions["ltol"] = 1e-5
noptions["giter"] = 21

def ffunc(x, func):
    return func(np.array([x]))[0]

def gfunc(x, func):

    g = np.zeros(dim)
    for i in range(dim):
        g[i] = func(np.array([x]), i)[0]

    return g

lconvx = []
lconvf = []
nl = 0
gconvx = []
gconvf = []
ng = 0

def lcallback(xi):
    global nl
    lconvx.append(nl)
    lconvf.append(ffunc(xi, func))
    nl += 1 

def gcallback(xi, convergence=None):
    global ng
    gconvx.append(ng)
    gconvf.append(ffunc(xi, func))
    ng += noptions["gpop"]

x0 = np.array([-1.,-1.])
args = (func,)
bounds = func.xlimits
# import pdb; pdb.set_trace()
### SLSQP
resultsl = optimize(ffunc, args=args, bounds=bounds, type="local", jac=gfunc, x0=x0, callback=lcallback, options=noptions)




### GA
resultsg = optimize(ffunc, args=args, bounds=bounds, type="global", callback=gcallback, options=noptions)


ax = plt.gca()
plt.plot(lconvx, lconvf, "b-", label=f'SLSQP')
plt.plot(gconvx, gconvf, 'r-', label='GA')
plt.yscale('log')

plt.xlabel(r"Number of Function Evaluations")
plt.ylabel(r"f(x)")
plt.title("Rosenbrock Optimization")
plt.ylim(1e-5, 1.)
plt.grid()

plt.legend()
plt.savefig(f"gradoptex.pdf", bbox_inches="tight")
plt.clf()

import pdb; pdb.set_trace()



