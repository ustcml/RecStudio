import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr

r"""
LorentzFM
######################

Paper Reference:
    Learning Feature Interactions with Lorentzian Factorization Machine (AAAI'20)
    https://arxiv.org/pdf/1911.09821
"""

class LorentzFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.lfm = nn.Sequential(OrderedDict([
                        ("embeddings", 
                            ctr.Embeddings(fields=self.fields, embed_dim=self.embed_dim, data=train_data)),
                        ("triangle_pooling_layer", 
                            ctr.TrianglePoolingLayer((len(self.fields) - 1)))
                    ]))

    def score(self, batch):
        lfm_score = self.lfm(batch)
        return {'score' : lfm_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
