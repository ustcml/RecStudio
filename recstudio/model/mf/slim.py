from recstudio.model.mf.ease import EASE
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
import scipy.sparse as sp
import numpy as np
import torch
import warnings
class SLIM(EASE):
    def training_epoch(self, nepoch):
        data, iscombine = self.current_epoch_trainloaders(nepoch)
        X = data['user_item_matrix'].tolil()
        model = ElasticNet(
            alpha=self.config.get('alpha'),
            l1_ratio=self.config.get('l1_ratio'),
            positive=self.config.get('positive_only'),
            fit_intercept=False,
            copy_X=False,
            precompute=True,
            selection='random',
            max_iter=100,
            tol=1e-4
        )
        item_coeffs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for j in range(X.shape[1]):
                r = X[:, j]
                X[:, j] = 0
                model.fit(X, r.A)
                item_coeffs.append(model.sparse_coef_)
                X[:, j] = r
        B = sp.vstack(item_coeffs).T
        self.item_vector = B[:, 1:]
        self.query_encoder.user = X
        return torch.tensor(0.)