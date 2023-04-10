import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr

r"""
FFM
######################

Paper Reference:
    Field-aware Factorization Machines for CTR Prediction (RecSys'16)
    https://dl.acm.org/doi/10.1145/2959100.2959134
"""

class FFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        num_fields = len(self.fields) - 1
        self.ffm = nn.Sequential(
                        OrderedDict([
                            ("embedding",
                                ctr.Embeddings(
                                    self.fields, 
                                    self.embed_dim * (num_fields - 1), 
                                    train_data)),
                            ("ffm_layer",
                                ctr.FieldAwareFMLayer(
                                    num_fields
                                ))
                        ]))
        
    def score(self, batch):
        lr_score = self.linear(batch)
        ffm_score = self.ffm(batch)
        return {'score' : lr_score + ffm_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
