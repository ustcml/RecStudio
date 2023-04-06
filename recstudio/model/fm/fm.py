import torch
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr


class FM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.fm = torch.nn.Sequential(OrderedDict([
            ("embeddings", ctr.Embeddings(
                fields=self.fields,
                embed_dim=self.embed_dim,
                data=train_data)),
            ("fm_layer", ctr.FMLayer(reduction='sum')),
        ]))
        self.linear = ctr.LinearLayer(self.fields, train_data)

    def score(self, batch):
        fm_score = self.fm(batch)
        lr_score = self.linear(batch)
        return {'score' : fm_score + lr_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
