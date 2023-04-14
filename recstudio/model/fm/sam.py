import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, LambdaLayer

r"""
SAM
######################

Paper Reference:
    Looking at CTR Prediction Again: Is Attention All You Need? (SIGIR'21)
    https://dl.acm.org/doi/10.1145/3404835.3462936
"""

class SAM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        num_fields = len(self.fields) - 1
        model_config = self.config['model']
        fi = model_config['interaction_type'].lower()
        self.sam = nn.Sequential(OrderedDict([
                        ('embedding',
                            ctr.Embeddings(self.fields, self.embed_dim, train_data)),
                        ('interaction',
                            ctr.SAMFeatureInteraction(
                                fi,
                                self.embed_dim,
                                num_fields,
                                model_config['dropout']))
                    ]))
        if fi == 'sam1':
            self.sam.add_module('agg', nn.Flatten(start_dim=1))
            self.sam.add_module('fc', nn.Linear(num_fields * self.embed_dim, 1))
        elif fi in ['sam2a', 'sam2e']:
            self.sam.add_module('agg', nn.Flatten(start_dim=1))
            self.sam.add_module('fc', nn.Linear(num_fields * num_fields * self.embed_dim, 1))
        else:
            self.sam.add_module('agg', nn.Sequential(
                                            LambdaLayer(lambda x: x.transpose(1, 2)),
                                            nn.Linear(num_fields, 1, bias=False),
                                            LambdaLayer(lambda x: x.sum(-1))))
            self.sam.add_module('fc', nn.Linear(self.embed_dim, 1))
            
            
    def score(self, batch):
        score = self.sam(batch).squeeze(-1)
        return{'score': score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
