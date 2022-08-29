from recstudio.model.mf.ease import EASE
from recstudio.model.basemodel import Recommender
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
import scipy.sparse as sp
import torch
import warnings


class SLIM(EASE):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('SLIM')
        parent_parser.add_argument("--knn", type=int, default=100, help='k for K-nearest neighbor')
        parent_parser.add_argument("--alpha", type=float, default=1.0, help='alpha coef')
        parent_parser.add_argument("--l1_ratio", type=float, default=0.1, help='coef for L1 regularization')
        parent_parser.add_argument("--positive_only", action='store_true', default=True, help='positive only flag')
        return parent_parser

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
