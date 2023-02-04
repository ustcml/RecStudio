from collections import OrderedDict

import torch
from recstudio.data import TripletDataset

from .. import basemodel, loss_func, scorer
from ..module import LambdaLayer, MLPModule, ctr


class DSSM(basemodel.BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DSSM')
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[256, 256, 256], help='mlp layer size')
        parent_parser.add_argument("--activation", type=str, default='tanh', help='activation function name')
        parent_parser.add_argument("--dropout", type=float, default=0.3, help='dropout rate for MLP')
        parent_parser.add_argument("--batch_norm", action='store_true', help='whether to use batch norm')
        parent_parser.add_argument("--negative_count", type=int, default=1, help='negative sampling numbers')
        return parent_parser

    def _set_data_field(self, data):
        data.use_field = data.field

    def _get_dataset_class():
        return TripletDataset

    def _get_query_encoder(self, train_data):
        if len(self.query_fields) == 1 and list(self.query_fields)[0] == self.fuid:
            embedding = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
            mlp_input_dim = self.embed_dim
        else:
            embedding = ctr.Embeddings(
                fields=self.query_fields,
                data=train_data,
                embed_dim=self.embed_dim)
            mlp_input_dim = embedding.num_features * self.embed_dim
        mlp = MLPModule(
            [mlp_input_dim] + self.config['mlp_layer'],
            dropout=self.config['dropout'], activation_func=self.config['activation'],
            batch_norm=self.config['batch_norm'])
        return torch.nn.Sequential(
            OrderedDict(
                {'embedding': embedding,
                 'flatten': LambdaLayer(lambda x: x.view(x.size(0), -1)),
                 'MLP': mlp}))

    def _get_item_encoder(self, train_data):
        if len(self.item_fields) == 1 and list(self.item_fields)[0] == self.fiid:
            embedding = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
            mlp_input_dim = self.embed_dim
            flatten_layer = LambdaLayer(lambda x: x)
        else:
            embedding = ctr.Embeddings(
                fields=self.item_fields,
                data=train_data,
                embed_dim=self.embed_dim,
            )
            mlp_input_dim = embedding.num_features * self.embed_dim
            flatten_layer = LambdaLayer(lambda x: x.view(*x.shape[: -2], -1))

        mlp = MLPModule(
            [mlp_input_dim] + self.config['mlp_layer'],
            activation_func=self.config['activation'],
            dropout=self.config['dropout'],
            batch_norm=self.config['batch_norm'])
        return torch.nn.Sequential(
            OrderedDict(
                {'embedding': embedding,
                 'flatten': flatten_layer,
                 'MLP': mlp}))

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.BinaryCrossEntropyLoss()
