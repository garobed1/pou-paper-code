
header = "getting_adapt_plots"
path = None
skip_LHS = True 
LHS_batch = 7
runs_per_proc = 1

# Problem Conditions
prob  = "fuhgp3"    #problem
dim = 1     #problem dimension


# Surrogate Settings
stype = "pouhess"   #surrogate type

### FOR POU HESS
rtype =  "hess"
opt = 'L-BFGS-B' #'SLSQP'#
local = False
gopt = 'ga' #'brute'
localswitch = True

### FOR POU SFCVT
# rtype =  "pousfcvt"
# opt = 'SLSQP' #for SFCVT constraint
# local = True
# localswitch = True

### FOR REGULAR SFCVT
# rtype =  "sfcvt"
# opt = 'SLSQP' #for SFCVT constraint, 
# local = False
# localswitch = False #fully global optimizer

corr  = "squar_exp"  #kriging correlation
poly  = "linear"    #kriging regression
delta_x = 1e-4 #1e-7
extra = dim           #gek extra points
t0 = [1e-0]
tb = [1e-6, 2e+1]
rscale = 5.5
rho = 10           #POU parameter

# Adaptive Sampling Settings
nt0  = 10       #initial design size
ntr = 10      #number of points to add
ntot = nt0 + ntr    #total number of points
batch = 1#dim*2        #batch size for refinement, as a percentage of ntr
Nerr = 5000*dim       #number of test points to evaluate the error
pperb = batch
pperbk = int(ntr/LHS_batch)
mstarttype = 2            # 0: No multistart
                          # 1: Start at the best out of a number of samples
                          # 2: Perform multiple optimizations
if(mstarttype == 1):   
    multistart = 50*dim
if(mstarttype == 2):
    multistart = 5*dim

if(pperb == 0):
    pperb = 1

# Other Refinement Settings
neval = 1+(dim+2) #applies to both criteria and POU model
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = True
bpen = False
obj = "inv"
nscale = 10.0 #1.0 for 2D
nmatch = dim

rc_print = False
