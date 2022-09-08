import torch
from collections import OrderedDict
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, LambdaLayer, HStackLayer
from recstudio.data.dataset import MFDataset


class FM(BaseRanker):

    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        fm = torch.nn.Sequential(OrderedDict({
            "embeddings": ctr.Embeddings(
                fields=self.fields,
                embed_dim=self.embed_dim,
                data=train_data
            ),
            "fm_layer": ctr.FMLayer(reduction='sum'),
        }))
        return torch.nn.Sequential(
            HStackLayer(OrderedDict({
                "fm": fm,
                "linear": ctr.LinearLayer(self.fields, train_data)
            })),
            LambdaLayer(lambda x: x[0]+x[1])
        )

    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
