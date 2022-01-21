import numpy as np
import matplotlib.pyplot as plt
import pougrad

from smt.sampling_methods import LHS
from smt.problems import LpNorm
from smt.surrogate_models import gekpls

dim = 100

trueFunc = LpNorm(ndim=dim, order = 1)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits)

ntr = 100
nte = 20

tr = sampling(ntr)
#tr = np.random.rand(ntr,dim)*2 - 1
# tr = np.zeros([ntr,dim])
# tr[:,0] = np.linspace(-10,10,ntr)
gr = np.zeros([ntr,dim])

te = sampling(nte)
#te = np.random.rand(nte, dim)*2 - 1
#te = np.sort(te,0)
fe = np.zeros(nte)

fr = trueFunc(tr)

for i in range(dim):
    gr[:,i:i+1] = trueFunc(tr,i)
import pdb; pdb.set_trace()
fe = trueFunc(te)
pou = pougrad.POUSurrogate(tr,fr,gr,0.1)
gek = gekpls.GEKPLS(theta0=[1e-2], xlimits=trueFunc.xlimits)
gek.set_training_values(tr, fr)
for i in range(dim):
    gek.set_training_derivatives(tr, gr[:,i:i+1], i)
gek.train()

# test the surrogate
fesp = np.zeros(nte)
fesg = np.zeros(nte)

for i in range(nte):
    fesp[i] = pou.eval(te[i])
    fesg[i] = gek.predict_values(te[i:i+1,:])

errp = fesp - fe[:,0]
errg = fesg - fe[:,0]

print(errp)
print(errg)

#import pdb; pdb.set_trace()

# plt.plot(tr[:,0],fr[:,0])
# plt.plot(te[:,0],fes,"o")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig("sphereplot.png")
