import torch
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, LambdaLayer
from recstudio.data.dataset import MFDataset


class FM(BaseRanker):
    # def _init_model(self, train_data):
    #     super()._init_model(train_data)


    def _get_dataset_class(self):
        return MFDataset

    def _get_scorer(self, train_data):
        embedding = ctr.Embeddings(
            self.fields, 
            train_data.field2type, 
            {f: train_data.num_values(f) for f in self.fields}, 
            self.embed_dim, 
            train_data.frating)
        return torch.nn.Sequential(
            embedding,
            ctr.FMLayer(),
            LambdaLayer(lambda x: x.sum(-1))
        )

    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss(reduction='mean')