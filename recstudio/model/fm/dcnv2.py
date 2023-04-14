import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule

r"""
DCNv2
######################

Paper Reference:
    DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems (WWW'21)
    https://dl.acm.org/doi/10.1145/3442381.3450078
"""

class DCNv2(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_fields = self.embedding.num_features
        model_config = self.config['model']
        if model_config['low_rank'] is None:
            self.cross_net = ctr.CrossNetworkV2(num_fields * self.embed_dim, model_config['num_layers'])
        else:
            self.cross_net = ctr.CrossNetworkMix(num_fields * self.embed_dim, model_config['num_layers'], 
                                                model_config['low_rank'], model_config['num_experts'],
                                                model_config['cross_activation'])
            
        if model_config['combination'].lower() == 'parallel':
            self.mlp = MLPModule(
                        [num_fields * self.embed_dim] + model_config['mlp_layer'],
                        model_config['activation'],
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'])
            self.fc = nn.Linear(num_fields*self.embed_dim + model_config['mlp_layer'][-1], 1)
        elif model_config['combination'].lower() == 'stacked':
            self.mlp = MLPModule(
                        [num_fields * self.embed_dim] + model_config['mlp_layer'] + [1],
                        model_config['activation'],
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'],
                        last_activation=False,
                        last_bn=False)
        else:
            raise ValueError(f'Expect combination to be `parallel`|`stacked`, but got {model_config["combination"]}.')

    def score(self, batch):
        emb = self.embedding(batch).flatten(1)
        cross_out = self.cross_net(emb)
        if self.config['model']['combination'].lower() == 'parallel':
            deep_out = self.mlp(emb)
            score = self.fc(torch.cat([deep_out, cross_out], -1)).squeeze(-1)
        else:
            deep_out = self.mlp(cross_out)
            score = deep_out.squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
