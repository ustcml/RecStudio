import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, LambdaLayer

r"""
DIFM
######################

Paper Reference:
    A Dual Input-aware Factorization Machine for CTR Prediction (IJCAI'20)
    https://dl.acm.org/doi/10.5555/3491440.3491874
"""

class DIFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        num_fields = self.embedding.num_features
        self.vec_wise_fen = nn.Sequential(
                                ctr.SelfAttentionInteractingLayer(
                                    self.embed_dim,
                                    model_config['n_head'],
                                    model_config['dropout'],
                                    layer_norm=model_config['layer_norm']),
                                LambdaLayer(lambda x: x.reshape(x.size(0), -1)),
                                nn.Linear(
                                    num_fields * self.embed_dim, 
                                    num_fields, 
                                    bias=False))
        self.bit_wise_fen = nn.Sequential(
                                ctr.MLPModule(
                                    [num_fields * self.embed_dim] + model_config['mlp_layer'],
                                    model_config['activation'],
                                    model_config['dropout'],
                                    batch_norm=model_config['batch_norm']),
                                nn.Linear(
                                    model_config['mlp_layer'][-1], 
                                    num_fields, 
                                    bias=False))
        self.fm = ctr.FMLayer(reduction='sum')
        
    def score(self, batch):
        emb = self.embedding(batch)
        m_vec = self.vec_wise_fen(emb)
        m_bit = self.bit_wise_fen(emb.flatten(1))
        weight = m_vec + m_bit
        lr_score = (super(ctr.LinearLayer, self.linear).forward(batch).squeeze(-1) * weight).sum(-1) + self.linear.bias
        fm_score = self.fm(emb * weight.unsqueeze(-1))
        return {'score' : lr_score + fm_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
