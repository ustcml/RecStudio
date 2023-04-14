import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
CCPM
######################

Paper Reference:
    A Convolutional Click Prediction Model (CIKM'15)
    https://dl.acm.org/doi/10.1145/2806416.2806603
"""

class CCPM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        model_config = self.config['model']
        num_fields = len(self.fields) - 1
        self.conv = nn.Sequential(
                        OrderedDict([
                            ("embeddings", 
                                ctr.Embeddings(
                                    self.fields, 
                                    self.embed_dim, 
                                    train_data)),
                            ("conv_layer", 
                                ctr.ConvLayer(
                                    num_fields,
                                    channels=model_config['channels'],
                                    heights=model_config['heights']))
                        ]))                               
        self.mlp = MLPModule(
                    [3 * self.embed_dim * model_config['channels'][-1]] + model_config['mlp_layer'] + [1],
                    model_config['activation'], 
                    model_config['dropout'],
                    last_activation=False,
                    last_bn=False)

    def score(self, batch):
        conv_out = self.conv(batch)
        score = self.mlp(conv_out.flatten(1)).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
