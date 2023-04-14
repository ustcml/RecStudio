
import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule

r"""
FiGNN
######################

Paper Reference:
    Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction (CIKM'19)
    https://doi.org/10.1145/3357384.3357951
"""

class FiGNN(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_fields = self.embedding.num_features
        self.gnn = nn.Sequential(OrderedDict([
                        ('self_attn', 
                            ctr.SelfAttentionInteractingLayer(
                                self.embed_dim,
                                model_config['n_head'],
                                model_config['dropout'],
                                residual=True,
                                residual_project=False,
                                layer_norm=model_config['layer_norm'])),
                        ('fignn', 
                            ctr.FiGNNLayer(
                                num_fields,
                                self.embed_dim,
                                model_config['num_gnn_layers']))
                    ]))
        self.attn_pred = nn.ModuleDict({
                            'mlp1': nn.Linear(self.embed_dim, 1),
                            'mlp2': nn.Linear(num_fields * self.embed_dim, num_fields)
                        })
        if model_config['deep']:
            self.mlp = MLPModule(
                    [self.embedding.num_features * self.embed_dim] + model_config['mlp_layer'] + [1],
                    model_config['activation'], 
                    model_config['dropout'],
                    last_activation=False, 
                    last_bn=False)
            
    def score(self, batch):
        emb = self.embedding(batch)
        gnn_out = self.gnn(emb)
        gnn_score = (self.attn_pred['mlp2'](gnn_out.flatten(1)) * \
                    self.attn_pred['mlp1'](gnn_out).squeeze(-1)).sum(-1)
        if self.config['model']['deep']:
            mlp_score = self.mlp(emb.flatten(1)).squeeze(-1)
            return {'score' : gnn_score + mlp_score}
        else:
            return{'score': gnn_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
