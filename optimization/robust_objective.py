import numpy as np

from utils.error import stat_comp



class RobustQuantity():
    """
    Base class that defines a generic robust quantity of interest. Provides
    the function handles that get minimized/evaluated for constrants, along 
    with handles for their gradients
    
    Flag as an objective or a constraint

    Types:
        "musigma" (default): Mean + \eta \sqrt{Variance}
        "failprob": 

    Types are implemented as derived classes

    """

    def __init__(self, **kwargs):
        """
        Initialize attributes
        """
        self.name = ''
        self.pathname = None
        self.comm = None

        self.type = None

        self.options = OptionsDictionary(parent_name=type(self).__name__)

        self.options.declare('assembled_jac_type', values=['csc', 'dense'], default='csc',
                             desc='Linear solver(s) in this group or implicit component, '
                                  'if using an assembled jacobian, will use this type.')


        self._declare_options()
        self.initialize()

        self.options.update(kwargs)


    def _declare_options(self):
        self.options.declare('opt_type', default="obj", values=["obj", "con"]
                             desc='objective or constraint')    

    def initialize(self):
        pass

    def func(self, xd)
        pass

    def grad(self, xd)
        pass


class MeanAndVar(RobustQuantity):

    """
    Quantity which is must (1-\eta)\mu + \eta\sigma

    """

    def _declare_options(self):
        
        super()._declare_options()

        self.options.declare('eta_val', types=float, default=0.5,
                             desc='')