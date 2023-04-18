
from smt.surrogate_models import KRG
import numpy as np

class GEK1D(KRG):
    name = "GEK1D"

    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        declare("delta_x", 1e-4, types=(int, float), desc="Step used in the FOTA")
        declare(
            "xlimits",
            types=np.ndarray,
            desc="Lower/upper bounds in each dimension - ndarray [nx, 2]",
        )
        self.supports["training_derivatives"] = True


    def _compute_pls(self, X, y):
        pts = self.training_points
        xlimits = self.options["xlimits"]
        delta_x = self.options["delta_x"]
        if 0 in self.training_points[None]: 
            nt, dim = X.shape
            XX = np.empty(shape=(0, dim))
            yy = np.empty(shape=(0, y.shape[1]))
            for i in range(nt):
                # Add additional points
                XX = np.vstack((XX, X[i, :]))
                XX[-1, 0] += delta_x * (xlimits[0, 1] - xlimits[0, 0])
                yy = np.vstack((yy, y[i]))
                yy[-1] += (
                    pts[None][1 + 0][1][i]
                    * delta_x
                    * (xlimits[0, 1] - xlimits[0, 0])
                )
            self.nt *= 2
            X = np.vstack((X, XX))
            y = np.vstack((y, yy))

        return X, y