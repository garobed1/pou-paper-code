import numpy as np
from loocvapprox import looCV

from smt.problems import Sphere
from smt.surrogate_models import gekpls
from smt.sampling_methods import LHS

dim = 3

trueFunc = Sphere(ndim=dim)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits)

ntr = 20
nte = 10

tr = sampling(ntr)
gr = np.zeros([ntr,dim])

te = sampling(nte)
fe = np.zeros(nte)

fr = trueFunc(tr)

for i in range(dim):
    gr[:,i:i+1] = trueFunc(tr,i)

fe = trueFunc(te)
gek = gekpls.GEKPLS(theta0=[1e-2], xlimits=trueFunc.xlimits)
gek.set_training_values(tr, fr)
for i in range(dim):
    gek.set_training_derivatives(tr, gr[:,i:i+1], i)
gek.train()

# Test if loocv works
criteria = looCV(gek, approx=False)

tloo = criteria.evaluate(te)

