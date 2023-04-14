import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
InterHAT
######################

Paper Reference:
    Interpretable Click-Through Rate Prediction through Hierarchical Attention (WSDM'20)
    https://dl.acm.org/doi/10.1145/3336191.3371785
"""

class InterHAT(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        self.trm = nn.TransformerEncoderLayer(
                        self.embed_dim, model_config['n_head'], 
                        model_config['feedforward_dim'], 
                        model_config['dropout'],
                        model_config['activation'],
                        batch_first=True)
        self.aggs = nn.ModuleList([
                        ctr.AttentionalAggregation(
                            self.embed_dim, 
                            model_config['aggregation_dim']) 
                        for _ in range(model_config['order'] + 1)
                    ])
        self.mlp = MLPModule([self.embed_dim] + model_config['mlp_layer'] + [1],
                        model_config['activation'], 
                        model_config['dropout'],
                        last_activation=False, 
                        last_bn=False
                    )

    def score(self, batch):
        emb = self.embedding(batch)
        xi = x1 = self.trm(emb)
        U = []
        for i, agg in enumerate(self.aggs[:-1]):
            ui = agg(xi, xi)
            U.append(ui)
            if i < self.config['model']['order']:
                xi = ui.unsqueeze(1) * x1 + xi
        U = torch.stack(U, dim=1)
        uf = self.aggs[-1](U, U)
        score = self.mlp(uf).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
