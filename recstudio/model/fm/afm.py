import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr

r"""
AFM
######################

Paper Reference:
    Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks (IJCAI'17)
    https://dl.acm.org/doi/10.5555/3172077.3172324
"""

class AFM(BaseRanker):
    
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("AFM")
        parent_parser.add_argument("--attention_dim", type=int, default=4, help="dimension for modeling attention")
        parent_parser.add_argument("--dropout", type=float, default=0.5, help="dropout probablity")
        return parent_parser

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        num_fields = len(self.fields) - 1
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.afm = nn.Sequential(
                    OrderedDict([
                        ("embeddings", 
                            ctr.Embeddings(
                                self.fields, 
                                self.embed_dim, 
                                train_data)),
                        ("afm_layer", 
                            ctr.AFMLayer(
                                self.embed_dim,
                                self.config['model']['attention_dim'],
                                num_fields,
                                self.config['model']['dropout']))
                    ]))

    def score(self, batch):
        lr_score = self.linear(batch)
        afm_score = self.afm(batch)
        return {'score' : lr_score + afm_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
