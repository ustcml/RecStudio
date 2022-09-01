import torch
from collections import OrderedDict
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, LambdaLayer, MLPModule, HStackLayer
from recstudio.data.dataset import MFDataset


class DeepFM(BaseRanker):
    # def _init_model(self, train_data):
    #     super()._init_model(train_data)

    @staticmethod
    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        embedding = ctr.Embeddings(
            self.fields,
            self.embed_dim,
            train_data)

        linear = ctr.LinearLayer(self.fields, train_data)

        return torch.nn.Sequential(
            HStackLayer(OrderedDict({
                'linear': linear,
                'fm_mlp': torch.nn.Sequential(
                    embedding,
                    HStackLayer(
                        ctr.FMLayer(reduction='sum'),
                        torch.nn.Sequential(
                            LambdaLayer(lambda x: x.view(x.size(0), -1)),
                            MLPModule([embedding.num_features*self.embed_dim]+self.config['mlp_layer'],
                                      self.config['activation'], self.config['dropout']),
                            torch.nn.Linear(self.config['mlp_layer'][-1], 1),
                            LambdaLayer(lambda x: x.squeeze(-1))
                        )
                    ),
                    LambdaLayer(lambda x: x[0]+x[1])
                )
            })),
            LambdaLayer(lambda x: x[0]+x[1])
        )

    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
