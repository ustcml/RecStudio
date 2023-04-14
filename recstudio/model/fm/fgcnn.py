import torch
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
FGCNN
######################

Paper Reference:
    Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction (WWW'19)
    https://dl.acm.org/doi/abs/10.1145/3308558.3313497
"""

class FGCNN(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        num_raw_fields = self.embedding.num_features
        model_config = self.config['model']
        self.fgcnn = ctr.FGCNNLayer(
                        num_raw_fields,
                        self.embed_dim,
                        model_config['channels'],
                        model_config['heights'],
                        model_config['pooling_sizes'],
                        model_config['recombine_channels'],
                        model_config['batch_norm'])  
        num_new_fields = sum([rc * oh for rc, oh in zip(model_config['recombine_channels'], self.fgcnn.out_height[1:])])
        num_total_fields = num_raw_fields + num_new_fields
        self.inner_product = ctr.InnerProductLayer(num_total_fields)
        mlp_in = num_total_fields * (num_total_fields - 1) // 2 + num_total_fields * self.embed_dim                       
        self.mlp = MLPModule(
                    [mlp_in] + model_config['mlp_layer'] + [1],
                    model_config['activation'], 
                    model_config['dropout'],
                    last_activation=False,
                    last_bn=False)

    def score(self, batch):
        raw_emb = self.embedding(batch)
        new_emb = self.fgcnn(raw_emb)
        comb_emb = torch.cat([raw_emb, new_emb], dim=1)
        inner_prod = self.inner_product(comb_emb)
        mlp_in = torch.cat([comb_emb.flatten(1), inner_prod], dim=1)
        score = self.mlp(mlp_in).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
