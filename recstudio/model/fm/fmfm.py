import torch
import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr

r"""
FmFM
######################

Paper Reference:
    FM^2: Field-matrixed Factorization Machines for Recommender Systems (WWW'21)
    https://dl.acm.org/doi/10.1145/3442381.3449930
"""

class FmFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_fields = self.embedding.num_features
        self.field_weight = nn.Parameter(torch.randn(num_fields*(num_fields - 1)//2, self.embed_dim, self.embed_dim))
        self.triu_index = nn.Parameter(
                            torch.triu_indices(num_fields, num_fields, offset=1), 
                            requires_grad=False)
        
    def score(self, batch):
        lr_score = self.linear(batch)
        emb = self.embedding(batch)
        emb0 = torch.index_select(emb, 1, self.triu_index[0])
        emb1 = torch.index_select(emb, 1, self.triu_index[1])
        fmfm_score = ((emb0.unsqueeze(-2) @ self.field_weight).squeeze(-2) * emb1).sum((-1, -2))
        return {'score' : lr_score + fmfm_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
