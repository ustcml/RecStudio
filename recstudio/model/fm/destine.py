import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
DESTINE
######################

Paper Reference:
    Disentangled Self-Attentive Neural Networks for Click-Through Rate Prediction (CIKM'21)
    https://dl.acm.org/doi/10.1145/3459637.3482088
"""

class DESTINE(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        if model_config['wide']:
            self.linear = ctr.LinearLayer(self.fields, train_data)
        if model_config['deep']:
            self.mlp = MLPModule([self.embedding.num_features * self.embed_dim] + model_config['mlp_layer'] + [1],
                            model_config['activation'], 
                            model_config['dropout'],
                            last_activation=False, 
                            last_bn=False
                        )
        
        if model_config['res_mode'].lower() == 'last_layer':
            self.res = nn.Linear(self.embed_dim, model_config['attention_dim'])
        elif model_config['res_mode'] is not None and model_config['res_mode'] != 'each_layer':
            raise ValueError(f'Expect res_mode to be `last_layer`|`each_layer`|None, but got {model_config["res_mode"]}.')
            
        self.dsa = nn.Sequential(*[
                        ctr.DisentangledSelfAttentionInteractingLayer(
                            self.embed_dim,
                            attention_dim=self.embed_dim if i == 0 else model_config['attention_dim'],
                            n_head=model_config['n_head'],
                            dropout=model_config['dropout'],
                            residual=(model_config['res_mode']=='each_layer'),
                            scale=model_config['scale'],
                            relu_before_att=model_config['relu_before_att'] if i == 0 else False,
                        )
                        for i in range(model_config['num_attention_layers'])])
        self.fc = nn.Linear(self.embedding.num_features * self.embed_dim, 1)

    def score(self, batch):
        emb = self.embedding(batch)
        attn_out = self.dsa(emb)
        if self.config['model']['res_mode'].lower() == 'last_layer':
            attn_out += self.res(emb)
        attn_out = attn_out.relu()
        attn_score = self.fc(attn_out.flatten(1)).squeeze(-1)
        score = attn_score
        if self.config['model']['wide']:
            lr_score = self.linear(batch)
            score += lr_score
        if self.config['model']['deep']:
            mlp_score = self.mlp(emb.flatten(1)).squeeze(-1)
            score += mlp_score
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
