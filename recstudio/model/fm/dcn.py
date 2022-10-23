import torch
from recstudio.data.dataset import MFDataset

from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule


class DCN(BaseRanker):

    def _get_dataset_class():
        return MFDataset

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_features = self.embedding.num_features
        mlp_layer = self.config['mlp_layer']
        self.cross_net = ctr.CrossNetwork(num_features * self.embed_dim,
                                          self.config['num_layers'])
        self.mlp = MLPModule(
                    [num_features * self.embed_dim] + mlp_layer,
                    dropout = self.config['dropout'],
                    batch_norm = self.config['batch_norm'])
        self.fc = torch.nn.Linear(num_features*self.embed_dim + mlp_layer[-1], 1)

    def score(self, batch):
        emb = self.embedding(batch)
        emb = emb.view(*emb.shape[:-2], -1)
        cross_out = self.cross_net(emb)
        deep_out = self.mlp(emb)
        score = self.fc(torch.cat([deep_out, cross_out], -1)).squeeze(-1)
        return score

    def _get_loss_func(self):
        return BCEWithLogitLoss(self.rating_threshold)
