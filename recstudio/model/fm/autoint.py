import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
AutoInt
######################

Paper Reference:
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (CIKM'19)
    https://dl.acm.org/doi/abs/10.1145/3357384.3357925
"""

class AutoInt(BaseRanker):
    
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("AutoInt")
        parent_parser.add_argument("--wide", type=bool, default=True, help="whether to use linear layer")
        parent_parser.add_argument("--deep", type=bool, default=True, help="whether to use mlp layer")
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[128,64], help="the MLP layer size")
        parent_parser.add_argument("--activation", type=str, default='relu', help="activation function")
        parent_parser.add_argument("--dropout", type=float, default=0.5, help="dropout probablity")
        parent_parser.add_argument("--attention_dim", type=int, default=64, help="dimension for modeling attention")
        parent_parser.add_argument("--num_attention_layers", type=int, default=3, help="number of attention layers")
        parent_parser.add_argument("--n_head", type=int, default=2, help="number of head for each multi-head attention layers")
        parent_parser.add_argument("--residual", type=bool, default=True, help="whether to use residual layer")
        parent_parser.add_argument("--residual_project", type=bool, default=True, help="whether to use residual project")
        parent_parser.add_argument("--layer_norm", type=bool, default=False, help="whether to use layer norm in attention")
        return parent_parser

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
        self.int = nn.Sequential(*[
                        ctr.InteractingLayer(
                            self.embed_dim,
                            n_head=model_config['n_head'],
                            dropout=model_config['dropout'],
                            residual=model_config['residual'],
                            residual_project=model_config['residual_project'],
                            layer_norm=model_config['layer_norm']
                        )
                        for _ in range(model_config['num_attention_layer'])])
        self.fc = nn.Linear(self.embedding.num_features * self.embed_dim, 1)

    def score(self, batch):
        emb = self.embedding(batch)
        attn_out = self.int(emb)
        int_score = self.fc(attn_out.reshape(attn_out.size(0), -1)).squeeze(-1)
        score = int_score
        if self.config['model']['wide']:
            lr_score = self.linear(batch)
            score += lr_score
        if self.config['model']['deep']:
            mlp_score = self.mlp(emb.view(emb.size(0), -1)).squeeze(-1)
            score += mlp_score
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
