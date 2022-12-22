"""
Interface to surrogate models available through the Surrogate Modeling Toolbox

SMT: M. A. Bouhlel and J. T. Hwang and N. Bartoli and R. Lafage and 
    J. Morlier and J. R. R. A. Martins

"""

import numpy as np

from smt.surrogate_models import KRG, KPLS, GEKPLS, LS
from surrogate.direct_gek import DGEK
from surrogate.pougrad import POUHessian

from collections import OrderedDict
from openmdao.surrogate_models.surrogate_model import SurrogateModel
from openmdao.utils.om_warnings import issue_warning, CacheWarning




_smt_models = OrderedDict([('KRG', KRG),
                              ('KPLS', KPLS),
                              ('GEKPLS', GEKPLS),
                              ('DGEK', DGEK),
                              ('LS', LS),
                              ('POU', POUHessian)])

#TODO: How to incorporate gradients?

class SMTSurrogate(SurrogateModel):

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Intention is that kwargs match those of SMT

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        # same approach as for the nearest neighbor models in OM
        if 'smt_type' in kwargs:
            self.options['smt_type'] = kwargs.pop('smt_type')

        self.n_dims = 0                 # number of independent
        self.n_samples = 0              # number of training points
        self.smt_model = None           # SMT model
        self.smt_init_args = kwargs

        # Initialize SMT object with kwargs
        self.smt_model = _smt_models[self.options['smt_type']](**self.smt_init_args)
        self.__name__ = "SMT_" + self.smt_model.name


    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('smt_type', default='KRG',
                             values=['KRG','KPLS','GEKPLS','DGEK','LS','POU'],
                             desc="Type of model to use")

    def train(self, x, y):
        """
        Train the surrogate model with the given set of inputs and outputs.

        Parameters
        ----------
        x : array-like
            Training input locations.
        y : array-like
            Model responses at given inputs.
        """
        super().train(x, y)
        self.smt_model.set_training_values(x, y)
        self.smt_model.train()

    def predict(self, x):
        """
        Calculate a predicted value of the response based on the current trained model.

        Parameters
        ----------
        x : array-like
            Point(s) at which the surrogate is evaluated.

        Returns
        -------
        float
            Predicted value.
        """
        super().predict(x)
        return self.smt_model.predict_values(x)

    def linearize(self, x, **kwargs):
        """
        Calculate the jacobian of the interpolant at the requested point.

        Parameters
        ----------
        x : array-like
            Point at which the surrogate Jacobian is evaluated.
        **kwargs : dict
            Additional keyword arguments passed to the interpolant.

        Returns
        -------
        ndarray
            Jacobian of surrogate output wrt inputs.
        """
        m, n = x.shape

        jac = np.zeros(m, n)

        for i in range(n):
            jac[:,i] = self.smt_model.predict_derivatives(x, i)
        if jac.shape[0] == 1 and len(jac.shape) > 2:
            return jac[0, ...]
        return jac