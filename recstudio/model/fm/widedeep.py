import torch
from collections import OrderedDict
from recstudio.data.dataset import MFDataset

from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, LambdaLayer, MLPModule, HStackLayer


class WideDeep(BaseRanker):

    def _get_dataset_class():
        return MFDataset

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        self.mlp = MLPModule(
                        [self.embedding.num_features*self.embed_dim]+self.config['mlp_layer']+[1],
                        activation_func = self.config['activation'],
                        dropout = self.config['dropout'],
                        batch_norm = self.config['batch_norm'],
                        last_activation = False, last_bn=False)

    def score(self, batch):
        wide_score = self.linear(batch)
        emb = self.embedding(batch)
        deep_score = self.mlp(emb.view(emb.size(0), -1)).squeeze(-1)
        return wide_score + deep_score

    def _get_loss_func(self):
        return BCEWithLogitLoss(threshold=self.rating_threshold)
