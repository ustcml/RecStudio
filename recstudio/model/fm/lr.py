from ..basemodel import BaseRanker
from ..module import ctr
from ..loss_func import BCEWithLogitLoss
from recstudio.data.dataset import TripletDataset


class LR(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)

    def _get_loss_func(self):
        return BCEWithLogitLoss()

    def score(self, batch):
        return {'score' : self.linear(batch)}
