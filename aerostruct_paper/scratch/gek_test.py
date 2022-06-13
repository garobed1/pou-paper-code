import sys, os
import copy
import pickle
from mpi4py import MPI
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from error import rmse, meane
from direct_gek import DGEK
from shock_problem import ImpingingShock
from example_problems import  QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, Peaks2D
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, TensorProduct
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from pougrad import POUSurrogate
from smt.sampling_methods import LHS
from scipy.stats import qmc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Perform adaptive sampling and estimate error
"""
prob  = "peaks"    #problem
plot  = -1

# Conditions
dim = 2      #problem dimension
corr  = "squar_exp" #kriging correlation
poly  = "quadratic"    #kriging regression 
ncomp = dim
extra = 2           #gek extra points
nt0 = 6
ntr = 80
batch = 10
t0 = [1e-2]
tb = [1e-6, 20]
tbg = [1e-6, 20]
Nerr = 5000       #number of test points to evaluate the error
dx = 1e-4
nruns = int((ntr-nt0)/batch)+1

# Problem Settings
alpha = 8.       #arctangent jump strength
if(prob == "arctan"):
    trueFunc = MultiDimJump(ndim=dim, alpha=alpha)
elif(prob == "arctantaper"):
    trueFunc = MultiDimJumpTaper(ndim=dim, alpha=alpha)
elif(prob == "rosenbrock"):
    trueFunc = Rosenbrock(ndim=dim)
elif(prob == "branin"):
    trueFunc = Branin(ndim=dim)
elif(prob == "sphere"):
    trueFunc = Sphere(ndim=dim)
elif(prob == "fuhgp8"):
    trueFunc = FuhgP8(ndim=dim)
elif(prob == "fuhgp9"):
    trueFunc = FuhgP9(ndim=dim)
elif(prob == "fuhgp10"):
    trueFunc = FuhgP10(ndim=dim)
elif(prob == "waterflow"):
    trueFunc = WaterFlow(ndim=dim)
elif(prob == "weldedbeam"):
    trueFunc = WeldedBeam(ndim=dim)
elif(prob == "robotarm"):
    trueFunc = RobotArm(ndim=dim)
elif(prob == "cantilever"):
    trueFunc = CantileverBeam(ndim=dim)
elif(prob == "hadamard"):
    trueFunc = QuadHadamard(ndim=dim)
elif(prob == "peaks"):
    trueFunc = Peaks2D(ndim=dim)
elif(prob == "tensorexp"):
    trueFunc = TensorProduct(ndim=dim, func="cos")
elif(prob == "shock"):
    xlimits = np.zeros([dim,2])
    xlimits[0,:] = [23., 27.]
    xlimits[1,:] = [0.36, 0.51]
    trueFunc = ImpingingShock(ndim=dim, input_bounds=xlimits, comm=MPI.COMM_SELF)
else:
    raise ValueError("Given problem not valid.")
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')

# Error
xtest = None 
ftest = None
testdata = None
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
testdata = [xtest, ftest]


nt = np.linspace(nt0, ntr, nruns, dtype=int)

xtrainK = []
ftrainK = []
gtrainK = []
for j in range(nruns):
    xtrainK.append(sampling(nt[j]))
    ftrainK.append(trueFunc(xtrainK[j]))
    gtrainK.append(np.zeros([nt[j],dim]))
    for i in range(dim):
        gtrainK[j][:,i:i+1] = trueFunc(xtrainK[j],i)

# add edge points
# nc = 2
# xc = np.zeros([nc, dim])
# xc[0] = np.array([xlimits[0,0]])
# xc[1] = np.array([xlimits[0,1]])
# fc = trueFunc(xc)
# gc = np.zeros([nc, dim])
# for i in range(dim):
#     gc[:,i:i+1] = trueFunc(xc,i)
# for j in range(nruns):
#     xtrainK[j] = np.append(xtrainK[j], xc, axis=0)
#     ftrainK[j] = np.append(ftrainK[j], fc, axis=0)
#     gtrainK[j] = np.append(gtrainK[j], gc, axis=0)

# add corner points
# nc = 2**dim
# xc = np.zeros([nc, dim])
# xc[0] = np.array([xlimits[0,0], xlimits[1,0]])
# xc[1] = np.array([xlimits[0,0], xlimits[1,1]])
# xc[2] = np.array([xlimits[0,1], xlimits[1,0]])
# xc[3] = np.array([xlimits[0,1], xlimits[1,1]])
# fc = trueFunc(xc)
# gc = np.zeros([nc, dim])
# for i in range(dim):
#     gc[:,i:i+1] = trueFunc(xc,i)
# xtrainK = np.append(xtrainK, xc, axis=0)
# ftrainK = np.append(ftrainK, fc, axis=0)
# gtrainK = np.append(gtrainK, gc, axis=0)


### FD CHECK
# h = 1e-6
# zero = 0.5*np.ones([1,2])
# step = 0.5*np.ones([1,2])
# step[0,0] += h
# ad1 = trueFunc(zero,0)
# fd1 = (trueFunc(step) - trueFunc(zero))/h
# step = 0.5*np.ones([1,2])
# step[0,1] += h
# ad2 = trueFunc(zero,1)
# fd2 = (trueFunc(step) - trueFunc(zero))/h
# import pdb; pdb.set_trace()


modeldge = DGEK(xlimits=xlimits)
modeldge.options.update({"theta0":t0})
modeldge.options.update({"theta_bounds":tb})
modeldge.options.update({"corr":corr})
modeldge.options.update({"poly":poly})
modeldge.options.update({"n_start":5})
modeldge.options.update({"print_prediction":False})
modeldge.set_training_values(xtrainK[0], ftrainK[0])
for i in range(dim):
    modeldge.set_training_derivatives(xtrainK[0], gtrainK[0][:,i:i+1], i)
modeldge.train()

import pdb; pdb.set_trace()

if(dim > 1):
    modelgek = GEKPLS(xlimits=xlimits)
    modelgek.options.update({"theta0":t0})
    modelgek.options.update({"theta_bounds":tbg})
    modelgek.options.update({"n_comp":ncomp})
    modelgek.options.update({"extra_points":extra})
    modelgek.options.update({"corr":corr})
    modelgek.options.update({"poly":poly})
    modelgek.options.update({"n_start":5})
    modelgek.options.update({"delta_x":dx})
    modelgek.options.update({"print_prediction":False})
else:
    modelgek = KRG()
    #modelgek.options.update({"hyper_opt":"TNC"})
    modelgek.options.update({"theta0":t0})
    modelgek.options.update({"theta_bounds":tbg})
    modelgek.options.update({"corr":corr})
    modelgek.options.update({"poly":poly})
    modelgek.options.update({"n_start":5})
    modelgek.options.update({"print_prediction":False})

modelkpl = KPLS()
#modelkpl.options.update({"hyper_opt":"TNC"})
modelkpl.options.update({"theta0":t0})
modelkpl.options.update({"theta_bounds":tb})
modelkpl.options.update({"n_comp":ncomp})
modelkpl.options.update({"corr":corr})
modelkpl.options.update({"poly":poly})
modelkpl.options.update({"n_start":5})
modelkpl.options.update({"print_prediction":False})

modelkrg = KRG()
#modelkrg.options.update({"hyper_opt":"TNC"})
modelkrg.options.update({"theta0":t0})
modelkrg.options.update({"theta_bounds":tb})
modelkrg.options.update({"corr":corr})
modelkrg.options.update({"poly":poly})
modelkrg.options.update({"n_start":5})
modelkrg.options.update({"print_prediction":False})

errgek = []
errkpl = []
errkrg = []
mgek = []
mkpl = []
mkrg = []
for j in range(nruns):
    if(dim > 1):
        modelgek.set_training_values(xtrainK[j], ftrainK[j])
        for i in range(dim):
            modelgek.set_training_derivatives(xtrainK[j], gtrainK[j][:,i:i+1], i)
        modelgek.train()
    else:
        nex = xtrainK[j].shape[0]
        xaug = np.zeros([nex, 1])
        faug = np.zeros([nex, 1])
        for k in range(nex):
            xaug[k] = xtrainK[j][k] + dx
            faug[k] = ftrainK[j][k] + dx*gtrainK[j][k]
        xtot = np.append(xtrainK[j], xaug, axis=0)
        ftot = np.append(ftrainK[j], faug, axis=0)
        modelgek.set_training_values(xtot, ftot)
        modelgek.train()
    mgek.append(copy.deepcopy(modelgek))

    modelkpl.set_training_values(xtrainK[j], ftrainK[j])
    modelkpl.train()
    mkpl.append(copy.deepcopy(modelkpl))

    modelkrg.set_training_values(xtrainK[j], ftrainK[j])
    modelkrg.train()
    errgek.append(rmse(modelgek, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    errkpl.append(rmse(modelkpl, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    errkrg.append(rmse(modelkrg, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    mkrg.append(copy.deepcopy(modelkrg))

print(errgek)
print(modelgek.optimal_theta)
print(errkpl)
print(modelkpl.optimal_theta)
print(errkrg)
print(modelkrg.optimal_theta)

plt.loglog(nt, errgek, "k-", label=f'GEK')
plt.loglog(nt, errkpl, 'b-', label='KPL')
plt.loglog(nt, errkrg, 'r-', label='KRG')
plt.xlabel("Number of samples")
plt.ylabel("NRMSE")
plt.grid()
plt.legend(loc=1)
plt.savefig(f"./gek_test_err.png", bbox_inches="tight")
plt.clf()


# # Plot points
if(dim == 1):  

    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)

    Zgek = np.zeros([ndir, ndir])
    Zkpl = np.zeros([ndir, ndir])
    Zkrg = np.zeros([ndir, ndir])
    Fgek = np.zeros([ndir, ndir])
    Fkpl = np.zeros([ndir, ndir])
    Fkrg = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        xi = np.zeros([1,1])
        xi[0] = x[i]
        TF[i] = trueFunc(xi)
        Fgek[i] = mgek[plot].predict_values(xi)
        Fkpl[i] = mkpl[plot].predict_values(xi)
        Fkrg[i] = mkrg[plot].predict_values(xi)
        Zgek[i] = abs(Fgek[i] - TF[i])
        Zkpl[i] = abs(Fkpl[i] - TF[i])
        Zkrg[i] = abs(Fkrg[i] - TF[i])

    # Plot Non-Adaptive Error
    plt.plot(x, TF, "-g")
    plt.plot(x, Fgek, "-k", label=f'GEK')
    plt.plot(x, Fkpl, "-b", label=f'KPL')
    plt.plot(x, Fkrg, "-r", label=f'KRG')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[plot][:,0], ftrainK[plot][:], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./gek_1dplot.png", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    plt.plot(x, Zgek, "-k", label=f'GEK')
    plt.plot(x, Zkpl, "-b", label=f'KPL')
    plt.plot(x, Zkrg, "-r", label=f'KRG')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$err$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[plot][:,0], np.zeros(xtrainK[plot].shape[0]), "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./gek_1derr.png", bbox_inches="tight")

    plt.clf()

# # Plot points
if(dim == 2):  
    # Plot Error Contour
    #Contour
    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

    X, Y = np.meshgrid(x, y)
    Za = np.zeros([ndir, ndir])
    Va = np.zeros([ndir, ndir])
    V0 = np.zeros([ndir, ndir])
    Zgek = np.zeros([ndir, ndir])
    Zkpl = np.zeros([ndir, ndir])
    Zkrg = np.zeros([ndir, ndir])
    Vk = np.zeros([ndir, ndir])
    Z0 = np.zeros([ndir, ndir])
    F  = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        for j in range(ndir):
            xi = np.zeros([1,2])
            xi[0,0] = x[i]
            xi[0,1] = y[j]
            TF[j,i] = trueFunc(xi)
            Zgek[j,i] = abs(modelgek.predict_values(xi) - TF[j,i])
            Zkpl[j,i] = abs(modelkpl.predict_values(xi) - TF[j,i])
            Zkrg[j,i] = abs(modelkrg.predict_values(xi) - TF[j,i])

    # Plot Non-Adaptive Error
    cs = plt.contour(X, Y, Zkpl, levels = 40)
    plt.colorbar(cs, aspect=20)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./kpl_errcona.png", bbox_inches="tight")

    plt.clf()

    plt.contour(X, Y, Zgek, levels = cs.levels)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./gek_errcona.png", bbox_inches="tight")

    plt.clf()

    

    plt.contour(X, Y, Zkrg, levels = cs.levels)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./krg_errcona.png", bbox_inches="tight")

    plt.clf()

