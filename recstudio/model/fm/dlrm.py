import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule, LambdaLayer

r"""
DLRM
######################

Paper Reference:
    Deep Learning Recommendation Model for Personalization and Recommendation Systems
    https://arxiv.org/abs/1906.00091
"""

class DLRM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        sparse_fields = {f for f in self.fields if train_data.field2type[f] != 'float'}
        dense_fields = {f for f in self.fields if train_data.field2type[f] == 'float' and f != self.frating}
        num_fields = len(sparse_fields) + int(len(dense_fields) > 0)
        self.embedding = ctr.Embeddings(sparse_fields, self.embed_dim, train_data)
        if len(dense_fields) > 0:
            self.bottom_mlp = MLPModule(
                                [len(dense_fields)] + model_config['bottom_mlp_layer'] + [self.embed_dim],
                                model_config['bottom_activation'], 
                                model_config['bottom_dropout'],
                                last_activation=False, 
                                last_bn=False)
        if model_config['op'].lower() == 'dot':
            self.interaction = ctr.InnerProductLayer(num_fields)
            top_mlp_in = num_fields * (num_fields - 1) // 2 + self.embed_dim * int(len(dense_fields) > 0)
        elif model_config['op'].lower() == 'cat':
            self.interaction = nn.Flatten(start_dim=1)
            top_mlp_in = num_fields * self.embed_dim
        elif model_config['op'].lower() == 'sum':
            self.interaction = LambdaLayer(lambda emb: emb.sum(1))
            top_mlp_in = self.embed_dim
        else:
            raise ValueError(f'Expect op to be `dot`|`cat`|`sum`, but got{model_config["op"]}.')
        self.top_mlp = MLPModule(
                        [top_mlp_in] + model_config['top_mlp_layer'] + [1],
                        model_config['top_activation'], 
                        model_config['top_dropout'],
                        last_activation=False, 
                        last_bn=False)
            
    def score(self, batch):
        emb = self.embedding(batch)
        dense_fields = {f for f in self.fields if f not in self.embedding.field2types and f != self.frating}
        if len(dense_fields) > 0:
            dense_values = torch.vstack([batch[f] for f in dense_fields]).t()
            if dense_values.dim() == 1:
                dense_values = dense_values.unsqueeze(-1)
            dense_emb = self.bottom_mlp(dense_values)
            emb = torch.cat([emb, dense_emb.unsqueeze(1)], dim=1)
        inter_out = self.interaction(emb)
        if self.config['model']['op'] == 'dot' and len(dense_fields) > 0:
            inter_out = torch.cat([inter_out, dense_emb], dim=-1)
        score = self.top_mlp(inter_out).squeeze(-1)
        return {'score': score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
