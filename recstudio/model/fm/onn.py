import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
ONN
######################

Paper Reference:
    Operation-aware Neural Networks for user response prediction (Neural Networks'20)
    https://dl.acm.org/doi/10.1016/j.neunet.2019.09.020
"""

class ONN(BaseRanker):
    
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("ONN")
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[128,64], help="the MLP layer size")
        parent_parser.add_argument("--activation", type=str, default='relu', help="activation function")
        parent_parser.add_argument("--dropout", type=float, default=0.2, help="dropout probablity")
        parent_parser.add_argument("--batch_norm", action='store_true', default=False, help="whether to use batch_norm")
        return parent_parser

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        num_fields = len(self.fields) - 1
        model_config = self.config['model']
        self.onn = nn.Sequential(
                        OrderedDict([
                            ("embedding",
                                ctr.Embeddings(
                                    self.fields, 
                                    self.embed_dim * num_fields, 
                                    train_data)),
                            ("ofm_layer",
                                ctr.OperationAwareFMLayer(
                                    num_fields
                                )),
                            ("mlp",
                                MLPModule(
                                    [num_fields * self.embed_dim + num_fields * (num_fields - 1) // 2] + model_config['mlp_layer'] + [1],
                                    model_config['activation'],
                                    model_config['dropout'],
                                    batch_norm=model_config['batch_norm'],
                                    last_activation=False, last_bn=True))
                        ]))
        

    def score(self, batch):
        onn_score = self.onn(batch).squeeze(-1)                    
        return {'score' : onn_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
