import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule, LambdaLayer

r"""
EDCN
######################

Paper Reference:
    Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models (CIKM'21)
    https://dl.acm.org/doi/10.1145/3459637.3481915
"""

class EDCN(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_fields = self.embedding.num_features
        model_config = self.config['model']
        self.cross = nn.ModuleList([
                            ctr.CrossInteraction(num_fields * self.embed_dim)
                            for _ in range(model_config['num_layers'])
                        ])
        self.mlp = nn.ModuleList([
                        MLPModule(
                            2 * [num_fields * self.embed_dim],
                            model_config['activation'],
                            model_config['dropout'],
                            batch_norm=model_config['batch_norm'])
                        for _ in range(model_config['num_layers'])   
                    ])
        self.bridge = nn.ModuleList([
                        ctr.BridgeLayer(
                            num_fields * self.embed_dim, 
                            model_config['bridge_type'])
                        for _ in range(model_config['num_layers'])
                    ])
        self.regulation = nn.ModuleList([
                            ctr.RegulationLayer(
                                num_fields, 
                                self.embed_dim, 
                                model_config['temperature'], 
                                model_config['batch_norm'])
                            for _ in range(model_config['num_layers'])
                        ])
        self.fc = torch.nn.Linear(3 * num_fields * self.embed_dim, 1)

    def score(self, batch):
        emb = self.embedding(batch)
        ci, di = self.regulation[0](emb.flatten(1))
        c0 = ci
        for i, (cross, deep, bridge) in enumerate(zip(self.cross, self.mlp, self.bridge)):
            ci = cross(c0, ci)
            di = deep(di)
            bi = bridge(ci, di)
            if i + 1 < self.config['model']['num_layers']:
                ci, di = self.regulation[i + 1](bi)
        score = self.fc(torch.cat([ci, di, bi], -1)).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
