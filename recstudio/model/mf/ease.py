from typing import OrderedDict

import numpy as np
import torch
from recstudio.data.dataset import TripletDataset
from recstudio.model import basemodel


class QueryEncoder(object):
    def __init__(self, user) -> None:
        self.user = user

    def __call__(self, batch):
        return self.user[batch, :]


class EASE(basemodel.BaseRetriever):

    def _get_dataset_class():
        return TripletDataset

    def _get_train_loaders(self, train_data):
        return {'user_item_matrix': train_data.get_graph(0, 'csr')[0]}

    def training_epoch(self, nepoch):
        if self.config['train']['gpu'] is not None:
            self.logger.warning("expecting EASE run on cpu while get gpu setting, automatically set gpu as None.")
            self.config['train']['gpu'] = None
        data, iscombine = self.current_epoch_trainloaders(nepoch)
        R = data['user_item_matrix']
        G = R.T @ R
        diagIndices = np.diag_indices_from(G)
        G[diagIndices] += self.config['train']['lambda']
        P = np.linalg.inv(G.todense())
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        self.item_vector = B[:, 1:]
        self.query_encoder.user = R
        return torch.tensor(np.linalg.norm(R-R*B, 'fro'))

    def _get_query_encoder(self, train_data):
        return QueryEncoder(None)

    def _get_score_func(self):
        def scorer(query, items):
            return torch.from_numpy((query @ items).A)
        return scorer

    def _get_loss_func(self):
        return None

    def _get_item_encoder(self, train_data):
        return None

    def _get_sampler(self, train_data):
        return None

    def _get_optimizers(self):
        return None

    def _get_item_vector(self):
        return self.item_vector

    def state_dict(self):
        return OrderedDict({
            'item_vector': getattr(self, 'item_vector', None),
            'query_encoder': getattr(self, 'query_encoder', None)
        })

    def load_state_dict(self, state_dict: OrderedDict):
        for k, v in state_dict.items():
            setattr(self, k, v)
