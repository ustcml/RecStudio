import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule

r"""
HFM
######################

Paper Reference:
    Holographic Factorization Machines for Recommendation (AAAI'19)
    https://dl.acm.org/doi/10.1609/aaai.v33i01.33015143
"""

class HFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_fields = self.embedding.num_features
        model_config = self.config['model']
        self.hfm = ctr.HolographicFMLayer(num_fields, model_config['op'])
        if model_config['deep']:
            self.mlp = MLPModule(
                    [num_fields * (num_fields - 1) // 2 * self.embed_dim] + model_config['mlp_layer'] + [1],
                    model_config['activation'], 
                    model_config['dropout'],
                    last_activation=False, 
                    last_bn=False)
        else:
            self.fc = nn.Linear(self.embed_dim, 1, bias=False)
            
    def score(self, batch):
        lr_score = self.linear(batch)
        emb = self.embedding(batch)
        hfm_out = self.hfm(emb)
        if self.config['model']['deep']:
            hfm_score = self.mlp(hfm_out.flatten(1)).squeeze(-1)
        else:
            hfm_score = self.fc(hfm_out.sum(1)).squeeze(-1)
        return{'score': lr_score + hfm_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
