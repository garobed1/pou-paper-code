# File containing dictionaries of default options for convenience

"""
method : string
    Optimization method
strategy : string
    Optimization method strategy
lmethod : string
    If not None, then refine the global optimizer output with the specified local optimizer
lstrategy : string
    Local optimization strategy, provide if lmethod is not None
giter : int
    Initial (global) optimizer iterations
gpop : int
    Population/Swarm size for global methods
liter : int
    Local optimizer iterations 
errorcheck : list, [xdata, fdata]
"""

DefaultOptOptions = {
    "method":"ga",
    "localswitch":True,
    "strategy":'best1bin', 
    "lmethod":None, 
    "lstrategy":None,
    "giter":100, 
    "gpop":15, 
    "gtol":0.01,
    "liter":100,
    "ltol":None,
    "errorcheck":None
}