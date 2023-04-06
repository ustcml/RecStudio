import torch
import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
AOANet
######################

Paper Reference:
    Architecture and Operation Adaptive Network for Online Recommendations (KDD'21)
    https://dl.acm.org/doi/10.1145/3447548.3467133
"""

class AOANet(BaseRanker):

    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("AOANet")
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[64,64], help="the MLP layer size")
        parent_parser.add_argument("--activation", type=str, default='relu', help="activation function")
        parent_parser.add_argument("--dropout", type=float, default=0.2, help="dropout probablity")
        parent_parser.add_argument("--num_subspaces", type=int, default=3, help="number of subspaces")
        parent_parser.add_argument("--num_interaction_layers", type=int, default=3, help="number of interaction layers")
        return parent_parser

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        self.mlp = MLPModule(
                    [self.embedding.num_features * self.embed_dim] + model_config['mlp_layer'],
                    model_config['activation'], 
                    model_config['dropout'],
                    last_activation=False, 
                    last_bn=False)
        self.gin = ctr.GeneralizedInteractionNet(
                        self.embedding.num_features,
                        self.embed_dim,
                        model_config['num_interaction_layers'],
                        model_config['num_subspaces'])
        self.fc = nn.Linear(model_config['mlp_layer'][-1] + model_config['num_subspaces'] * self.embed_dim, 1)
            

    def score(self, batch):
        emb = self.embedding(batch)
        mlp_out = self.mlp(emb.view(emb.size(0), -1))
        gin_out = self.gin(emb).reshape(emb.size(0), -1)
        score = self.fc(torch.cat([mlp_out, gin_out], dim=-1)).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
