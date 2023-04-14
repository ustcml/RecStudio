import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule, ResidualLayer

r"""
DeepCrossing
######################

Paper Reference:
    Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features (KDD'16)
    https://dl.acm.org/doi/10.1145/2939672.2939704
"""

class DeepCrossing(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        num_fields = len(self.fields) - 1
        self.dc = nn.Sequential(OrderedDict([
                    ("embedding",
                        ctr.Embeddings(self.fields, self.embed_dim, train_data)),
                    ("residuals",
                        nn.Sequential(*[
                            ResidualLayer(
                                MLPModule(
                                    [num_fields*self.embed_dim, hidden_dim, num_fields*self.embed_dim],
                                    model_config['activation'], 
                                    last_activation=False, last_bn=False
                                ),
                                num_fields,
                                self.embed_dim,
                                model_config['activation'],
                                model_config['dropout'],
                                model_config['batch_norm'],
                                model_config['layer_norm']
                            ) 
                            for hidden_dim in model_config['hidden_dims']]))
                ]))
        self.fc = nn.Linear(num_fields * self.embed_dim, 1)
    def score(self, batch):
        dc_out = self.dc(batch)
        score = self.fc(dc_out.flatten(1)).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
