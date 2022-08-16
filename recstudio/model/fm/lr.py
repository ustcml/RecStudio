import torch
from collections import OrderedDict
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, LambdaLayer, MLPModule, HStackLayer
from recstudio.data.dataset import MFDataset


class LR(BaseRanker):

    @staticmethod
    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        return ctr.LinearLayer(self.fields, train_data)

    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
