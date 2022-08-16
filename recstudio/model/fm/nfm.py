import torch
from collections import OrderedDict
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, LambdaLayer, MLPModule, HStackLayer
from recstudio.data.dataset import MFDataset


class NFM(BaseRanker):

    @staticmethod
    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        embeddings = ctr.Embeddings(
            self.fields,
            train_data.field2type,
            {f: train_data.num_values(f) for f in self.fields},
            embed_dim=self.embed_dim,
            rating_field=train_data.frating)

        linear = torch.nn.Sequential(OrderedDict({
            "embeddings": ctr.Embeddings(
                self.fields,
                train_data.field2type,
                {f: train_data.num_values(f) for f in self.fields},
                1,
                train_data.frating),
            "linear_layer": ctr.LinearLayer()
        }))
        return torch.nn.Sequential(
            HStackLayer(OrderedDict({
                "nfm": torch.nn.Sequential(
                    OrderedDict(
                        {'embeddings': embeddings, 'fm_layer': ctr.FMLayer(),
                         'bn': torch.nn.BatchNorm1d(self.embed_dim),
                         'mlp': MLPModule(
                            [self.embed_dim]+self.config['mlp_layer'],
                            activation_func=self.config['activation'],
                            dropout=self.config['dropout'],
                            batch_norm=self.config['batch_norm']),
                         'fc': torch.nn.Linear(self.config['mlp_layer'][-1], 1, bias=False),
                         'squeeze': LambdaLayer(lambda x: x.squeeze(-1))
                         })),
                "linear": linear})),
            LambdaLayer(lambda x: x[0]+x[1])
        )

    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
