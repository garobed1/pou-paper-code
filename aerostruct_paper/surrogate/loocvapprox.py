import numpy as np
import copy

from smt.utils.options_dictionary import OptionsDictionary
from smt.surrogate_models.surrogate_model import SurrogateModel

"""Base Class for Adaptive Sampling Criteria Functions"""
class ASCriteria():
    def __init__(self, model, **kwargs):
        """
        Constructor where a trained surrogate model is 

        Parameters
        ----------
        model : smt SurrogateModel object
            Surrogate model to perform the cross validation

        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.
        """
        self.options = OptionsDictionary()
        self.options.declare("approx", False, types=bool)
        self.options.update(kwargs)

        # copy the surrogate model object
        self.model = copy.deepcopy(model)

        # get the size of the training set
        kx = 0
        self.dim = self.model.training_points[None][kx][0].shape[1]
        self.ntr = self.model.training_points[None][kx][0].shape[0]

        self.initialize()

    def initialize(self):
        pass

    def evaluate(self, x):
        pass

    
"""
A Continuous Leave-One-Out Cross Validation function
"""
class looCV(ASCriteria):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        #TODO: Add more options for this
    
    def initialize(self):

        # Create a list of LOO surrogate models
        self.loosm = []
        for i in range(self.ntr):
            self.loosm.append(copy.deepcopy(self.model))
            self.loosm[i].options.update({"print_global":False})

            kx = 0

            # Give each LOO model its training data, and retrain if not approximating
            trx = self.loosm[i].training_points[None][kx][0]
            trf = self.loosm[i].training_points[None][kx][1]
            trg = []
            for j in range(self.dim):
                trg.append(self.loosm[i].training_points[None][j+1][1])
            #trg = self.loosm[i].training_points[None][kx+1][1] #TODO:make this optional
            trx = np.delete(trx, i, 0)
            trf = np.delete(trf, i, 0)
            for j in range(self.dim):
                trg[j] = np.delete(trg[j], i, 0)

            self.loosm[i].set_training_values(trx, trf)
            for j in range(self.dim):
                self.loosm[i].set_training_derivatives(trx, trg[j][:], j)

            if(self.options["approx"] == False):
                self.loosm[i].train()

    #TODO: This could be a variety of possible LOO-averaging functions
    def evaluate(self, x):
        
        # evaluate the point for the original model
        M = self.model.predict_values(x)

        # now evaluate the point for each LOO model and sum
        y = 0
        for i in range(self.ntr):
            Mm = self.loosm[i].predict_values(x)
            y += (1/self.ntr)*((M-Mm)**2)

        return np.sqrt(y)


