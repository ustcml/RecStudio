import torch
from collections import OrderedDict
from ..basemodel import BaseRanker
from ..module import ctr, LambdaLayer, MLPModule, HStackLayer
from ..loss_func import BCEWithLogitLoss
from recstudio.data.dataset import MFDataset


class LR(BaseRanker):

    @staticmethod
    def _get_dataset_class():
        return MFDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)

    def _get_loss_func(self):
        return BCEWithLogitLoss(threshold=self.rating_threshold)

    def score(self, batch):
        return self.linear(batch)
