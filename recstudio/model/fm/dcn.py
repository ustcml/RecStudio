import torch
from collections import OrderedDict
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, LambdaLayer, MLPModule, HStackLayer
from recstudio.data.dataset import MFDataset


class DCN(BaseRanker):

    @staticmethod
    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        embedding = ctr.Embeddings(
            self.fields,
            train_data.field2type,
            {f: train_data.num_values(f) for f in self.fields},
            self.embed_dim,
            train_data.frating)
        return torch.nn.Sequential(OrderedDict({
            'embedding': embedding,
            'flatten': LambdaLayer(lambda x: x.view(*x.shape[:-2], -1)),
            'cross_net': HStackLayer(
                ctr.CrossNetwork(embedding.num_features * self.embed_dim, self.config['num_layers']),
                MLPModule(
                    [embedding.num_features * self.embed_dim] + self.config['mlp_layer'],
                    dropout=self.config['dropout'],
                    batch_norm=self.config['batch_norm'])),
            'cat': LambdaLayer(lambda x: torch.cat(x, dim=-1)),
            'fc': torch.nn.Linear(embedding.num_features*self.embed_dim + self.config['mlp_layer'][-1], 1),
            'squeeze': LambdaLayer(lambda x: x.squeeze(-1))
        }))

    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
