import torch
from recstudio.data import dataset

from .. import basemodel, scorer, loss_func

class PMF(basemodel.BaseRetriever):

    @staticmethod
    def _get_dataset_class():
        return dataset.MFDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _init_parameter(self):
        super()._init_parameter()
        torch.nn.init.normal_(self.query_encoder.weight, mean=0, std=self.config['query_std'])
        torch.nn.init.normal_(self.item_encoder.weight, mean=0, std=self.config['item_std'])

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.SquareLoss()
