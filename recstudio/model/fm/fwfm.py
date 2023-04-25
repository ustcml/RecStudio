import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr

r"""
FwFM
######################

Paper Reference:
    Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising (WWW'18)
    https://dl.acm.org/doi/abs/10.1145/3178876.3186040
"""

class FwFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_fields = self.embedding.num_features
        self.fwfm = nn.Sequential(
                        OrderedDict([
                            ("fm_layer",
                                ctr.InnerProductLayer(
                                    num_fields)),
                            ("field_weighted",
                                nn.Linear(num_fields * (num_fields - 1) // 2, 1))
                        ]))
        if self.config['model']['linear_type'].lower() == 'lw':
            self.linear = ctr.LinearLayer(self.fields, train_data)
        elif self.config['model']['linear_type'].lower() == 'felv':
            self.linear_embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        elif self.config['model']['linear_type'].lower() == 'filv':
            self.linear = nn.Linear(num_fields * self.embed_dim, 1, bias=False)
        else:
            raise ValueError('Expect linear_type to be `lw`|`felv`|`filv`, '
                             f'but got {self.config["model"]["linear_type"]}.')
        
    def score(self, batch):
        emb = self.embedding(batch)
        if self.config['model']['linear_type'].lower() == 'lw':
            lr_score = self.linear(batch)
        elif self.config['model']['linear_type'].lower() == 'felv':
            lr_emb = self.linear_embedding(batch)
            lr_score = (lr_emb * emb).sum((1, 2))
        else:
            lr_score = self.linear(emb.flatten(1)).squeeze(-1)
        fwfm_score = self.fwfm(emb).squeeze(-1)
        return {'score' : lr_score + fwfm_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
