import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
FiBiNET
######################

Paper Reference:
    FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction (RecSys'19)
    https://dl.acm.org/doi/abs/10.1145/3298689.3347043
"""

class FiBiNET(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        num_fields = self.embedding.num_features
        self.senet = ctr.SqueezeExcitation(
                        num_fields, 
                        model_config['reduction_ratio'], 
                        model_config['excitation_activation'])
        self.bilinear = ctr.BilinearInteraction(
                            num_fields, 
                            self.embed_dim, 
                            model_config['bilinear_type'])
        if not model_config['shared_bilinear']:
            self.bilinear4se = ctr.BilinearInteraction(
                                num_fields, 
                                self.embed_dim, 
                                model_config['bilinear_type'])
        self.mlp = MLPModule(
                        [num_fields * (num_fields - 1) * self.embed_dim] + model_config['mlp_layer'] + [1],
                        model_config['activation'], 
                        model_config['dropout'],
                        last_activation=False, 
                        last_bn=False)

    def score(self, batch):
        lr_score = self.linear(batch)
        emb = self.embedding(batch)
        senet_emb = self.senet(emb)
        bilinear_ori = self.bilinear(emb)
        if self.config['model']['shared_bilinear']:
            bilinear_senet = self.bilinear(senet_emb)
        else:
            bilinear_senet = self.bilinear4se(senet_emb)
        comb = torch.cat([bilinear_ori, bilinear_senet], dim=1)
        mlp_score = self.mlp(comb.view(comb.size(0), -1)).squeeze(-1)
        return {'score' : lr_score + mlp_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
