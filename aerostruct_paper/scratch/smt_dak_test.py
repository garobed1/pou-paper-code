import sys, copy
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1,"../surrogate")


#import dakota.surrogates as DKS
from error import rmse, meane
from dakota_kriging import DakotaKriging
from example_problems import  QuadHadamard, MultiDimJump, MultiDimJumpTaper, FuhgP8, FuhgP9, FuhgP10, Peaks2D
from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam, TensorProduct
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG
from scipy.stats import qmc



prob  = "rosenbrock"    #problem
plot  = -1

# Conditions
dim = 2      #problem dimension
corr  = "squar_exp" #kriging correlation
poly  = "linear"    #kriging regression 
ncomp = dim
extra = 2           #gek extra points
nt0 = 64
ntr = 64
batch = 16
tval = 100.
t0 = [1e-2]
t0g = [tval]#[tval, tval]
tb = [1e-6, 100]
tbg = [tval, tval]#[tval, tval]
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
    trueFunc = TensorProduct(ndim=dim, func="gaussian")
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

# DAKOTA KRG
modeldk = DakotaKriging(xlimits=xlimits)
modeldk.options.update({"trend":poly})
modeldk.options.update({"optimization_method":"global"})
modeldk.options.update({"print_prediction":False})
#modeldk.options.update({"nugget":2e-14})
modeldk.set_training_values(xtrainK[0], ftrainK[0])
for i in range(dim):
    modeldk.set_training_derivatives(xtrainK[0], gtrainK[0][:,i:i+1], i)
modeldk.train()

# DAKOTA GEK
modeldgk = DakotaKriging(xlimits=xlimits)
modeldgk.options.update({"trend":poly})
modeldgk.options.update({"header":"py_dak_gek_"})
modeldgk.options.update({"use_derivatives":True})
modeldgk.options.update({"optimization_method":"global"})
modeldgk.options.update({"print_prediction":False})
#modeldgk.options.update({"nugget":2e-14})
modeldgk.set_training_values(xtrainK[0], ftrainK[0])
for i in range(dim):
    modeldgk.set_training_derivatives(xtrainK[0], gtrainK[0][:,i:i+1], i)
modeldgk.train()

modelkrg = KRG()
#modelkrg.options.update({"hyper_opt":"TNC"})
modelkrg.options.update({"theta0":t0})
modelkrg.options.update({"theta_bounds":tb})
modelkrg.options.update({"corr":corr})
modelkrg.options.update({"poly":poly})
modelkrg.options.update({"n_start":5})
modelkrg.options.update({"print_prediction":False})

errdkg = []
mdkrg = []
errkrg = []
mkrg = []
errdgk = []
mdgk = []
for j in range(nruns):
    modeldk.set_training_values(xtrainK[j], ftrainK[j])
    modeldk.train()
    mdkrg.append(modeldk)
    errdkg.append(rmse(modeldk, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    print("Dakota Kriging Done")

    modeldgk.set_training_values(xtrainK[j], ftrainK[j])
    for i in range(dim):
        modeldgk.set_training_derivatives(xtrainK[j], gtrainK[j][:,i:i+1], i)
    modeldgk.train()
    mdgk.append(modeldgk)
    errdgk.append(rmse(modeldgk, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    print("Dakota GEK Done")

    modelkrg.set_training_values(xtrainK[j], ftrainK[j])
    modelkrg.train()
    mkrg.append(copy.deepcopy(modelkrg))
    errkrg.append(rmse(modelkrg, trueFunc, N=Nerr, xdata=xtest, fdata=ftest))
    print("KRG Done")

print("Dakota Kriging")
print(errdkg)
print("SMT Kriging")
print(errkrg)
print(modelkrg.optimal_theta)
# print("GEK")
# print(errdgek)

# # print(errdge)
# # print(modeldge.optimal_theta)
plt.loglog(nt, errkrg, 'r-', label='SMT Kriging')
plt.loglog(nt, errdkg, 'c-', label='Dakota Kriging')
plt.xlabel("Number of samples")
plt.ylabel("NRMSE")
plt.grid()
plt.legend(loc=1)
plt.savefig(f"./dkrg_test_err.png", bbox_inches="tight")
plt.clf()



# # Plot points
if(dim == 1):  

    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)

    Zdkrg = np.zeros([ndir, ndir])
    Fdkrg = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        xi = np.zeros([1,1])
        xi[0] = x[i]
        TF[i] = trueFunc(xi)
        Fdkrg[i] = mdkrg[plot].value(xi)
        Zdkrg[i] = abs(Fdkrg[i] - TF[i])

    # Plot Non-Adaptive Error
    plt.plot(x, TF, "-g")
    plt.plot(x, Fdkrg, "-c", label=f'Dakota Kriging')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[plot][:,0], ftrainK[plot][:], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./dkrg_1dplot.png", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    plt.plot(x, Zdkrg, "-c", label=f'Dakota Kriging')

    plt.xlabel(r"$x$")
    plt.ylabel(r"$err$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[plot][:,0], np.zeros(xtrainK[plot].shape[0]), "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')

    plt.savefig(f"./dkrg_1derr.png", bbox_inches="tight")

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
    Zdkrg = np.zeros([ndir, ndir])
    Zgrb = np.zeros([ndir, ndir])
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
            Zkrg[j,i] = abs(mkrg[plot].predict_values(xi) - TF[j,i])
            Zdkrg[j,i] = abs(mdkrg[plot].predict_values(xi) - TF[j,i])
    # Plot Non-Adaptive Error
    cs = plt.contour(X, Y, Zkrg, levels = 40)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./krg_errcona.png", bbox_inches="tight")

    plt.clf()    

    plt.contour(X, Y, Zdkrg, levels = cs.levels)
    plt.colorbar(cs, )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(xtrainK[-1][:,0], xtrainK[-1][:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.savefig(f"./dkrg_errcona.png", bbox_inches="tight")

    plt.clf()

