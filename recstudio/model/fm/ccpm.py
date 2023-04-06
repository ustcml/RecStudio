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
    
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("CCPM")
        parent_parser.add_argument("--channels", type=int, nargs='+', default=[3,3], help="in channels for each convolution")
        parent_parser.add_argument("--heights", type=int, nargs='+', default=[6,5], help="heights for each convolution")
        parent_parser.add_argument("--conv_activation", type=str, default='tanh', help="activation function of convolution layer")
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[256], help="the MLP layer size")
        parent_parser.add_argument("--activation", type=str, default='relu', help="activation function")
        parent_parser.add_argument("--dropout", type=float, default=0.5, help="dropout probablity")
        return parent_parser

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
                                    heights=model_config['heights'],
                                    activation=model_config['conv_activation']))
                        ]))                               
        self.mlp = MLPModule(
                    [3 * self.embed_dim * model_config['channels'][-1]] + model_config['mlp_layer'] + [1],
                    model_config['activation'], 
                    model_config['dropout'],
                    last_activation=False,
                    last_bn=False)

    def score(self, batch):
        conv_out = self.conv(batch)
        score = self.mlp(conv_out.reshape(conv_out.size(0), -1)).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
