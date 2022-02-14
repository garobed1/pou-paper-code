import numpy as np
import matplotlib.pyplot as plt
from refinecriteria import looCV, HessianFit
from getxnew import getxnew, adaptivesampling
from defaults import DefaultOptOptions
from pougrad import POUSurrogate


from heaviside import Heaviside, Quad2D
from smt.problems import Sphere, LpNorm
from smt.surrogate_models import kpls, gekpls
from smt.sampling_methods import LHS

dim = 2
rho = 100

trueFunc = Heaviside(ndim=dim)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits)

nt0  = 4
dist = 0.01
ntr = 10
nte = 20

t0 = np.zeros([nt0,dim])
t0 = sampling(nt0)



g0 = np.zeros([nt0,dim])

# te = sampling(nte)
# fe = np.zeros(nte)

f0 = trueFunc(t0)



for i in range(dim):
    g0[:,i:i+1] = trueFunc(t0,i)

# fe = trueFunc(te)

# for "fake" gradient enhanced kriging in 1 dim
tk = np.zeros([2*nt0,dim])
fk = np.zeros([2*nt0,1])
for i in range(nt0):
    tk[2*i,[0]] = t0[i,[0]]-dist
    fk[2*i,[0]] = f0[i,[0]]-dist*g0[i,[0]]
    tk[2*i+1,[0]] = t0[i,[0]]+dist
    fk[2*i+1,[0]] = f0[i,[0]]+dist*g0[i,[0]]

#import pdb; pdb.set_trace()
#gek = kpls.KPLS()
gek = gekpls.GEKPLS(xlimits=xlimits)
#gek = POUSurrogate()
#gek.options.update({"rho":rho})
#gek.options.update({"poly":"linear"})
gek.options.update({"print_global":False})
gek.set_training_values(t0, f0)
#gek.set_training_values(np.append(t0, tk, axis=0), np.append(f0, fk, axis=0))
if(isinstance(gek, gekpls.GEKPLS) or isinstance(gek, POUSurrogate)):
    for i in range(dim):
        gek.set_training_derivatives(t0, g0[:,i:i+1], i)
gek.train()

ndir = 100

x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
xz = np.zeros([ndir,dim])
xz[:,0] = x
zt = trueFunc(xz)
zs = gek.predict_values(xz)

criteria = HessianFit(gek, improve=0.5) #looCV(gek, approx=False)

xzt = np.linspace(-1, 1, 50)
zlv = np.zeros(50)
for i in range(50):
    zlv[i] = criteria.evaluate(xzt[i])

bads = criteria.bads
bad_dirs = criteria.bad_dirs
#import pdb; pdb.set_trace()
plt.figure(0)
plt.plot(t0[:,0], t0[:,1], "o")
for i in range(bads.shape[0]):
    plt.plot(bads[i,0], bads[i,1], "ro")
    plt.plot([bads[i,0] - bad_dirs[i][0], bads[i,0] + bad_dirs[i][0]], \
                [bads[i,1] - bad_dirs[i][1], bads[i,1] + bad_dirs[i][1]], '--')
plt.savefig("funcplotHessinit.png")
plt.figure(1)


plt.plot(xzt, zlv, "b-")
plt.savefig("critplotHessinit.png")
import pdb; pdb.set_trace()


#import pdb; pdb.set_trace()
# Test if loocv works

#plt.plot(x, zlv, "b-")
#plt.savefig("asplot.png")
#import pdb; pdb.set_trace()

# Test if xnew works
tr = []#np.zeros([ntr,dim])
fr = []#np.zeros(ntr)
#gr = np.zeros([ntr,dim])

# for i in range(ntr):
#     x0 = 0.5
#     xnew = np.array([getxnew(criteria, x0, xlimits)])
#     t0 = np.append(t0, xnew, axis=0)
#     f0 = np.append(f0, trueFunc(xnew), axis=0)

#     gek.set_training_values(t0, f0)
#     # for i in range(dim):
#     #     gek.set_training_derivatives(t0, g0[:,i:i+1], i)
#     gek.train()

#     zs = gek.predict_values(x)

#     #replace criteria
#     criteria = looCV(gek, approx=False)
#     zlv = criteria.evaluate(x)


    
#     plt.figure(0)
#     plt.clf()
#     plt.plot(x, zt, "k-")
#     plt.plot(x, zs, "r-")
#     plt.plot(t0, f0, "o")
#     plt.savefig("asplot.png")

#     plt.figure(1)
#     plt.clf()
#     plt.plot(x, zlv, "b-")
#     plt.savefig("critplot.png")

#     import pdb; pdb.set_trace()

options = DefaultOptOptions
# options = None
options["localswitch"] = True
#import pdb; pdb.set_trace()
#gek.name = "GEKPLS"

#gek, criteria = adaptivesampling(trueFunc, gek, criteria, xlimits, ntr, options=options)

zs = gek.predict_values(x)
t0 = gek.training_points[None][0][0]
f0 = gek.training_points[None][0][1]
zlv = criteria.evaluate(x)

plt.figure(0)
plt.clf()
plt.plot(x, zt, "k-")
plt.plot(x, zs, "r-")
plt.plot(t0[0:nt0,0], f0[0:nt0,0], "bo")
plt.plot(t0[nt0:,0], f0[nt0:,0], "ro")
plt.savefig("asplotgek.png")
plt.figure(1)
plt.clf()
plt.plot(x, zlv, "b-")
plt.savefig("critplotgek.png")

# tloo = criteria.evaluate(te)

