import torch
import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
AFN
######################

Paper Reference:
    Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions (AAAI'20)
    https://arxiv.org/abs/1909.03276
"""

class AFN(BaseRanker):

    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("AFN")
        parent_parser.add_argument("--log_hidden_size", type=int, default=128, help="hidden size of logtransformer layer")
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[128,128], help="the MLP layer size")
        parent_parser.add_argument("--activation", type=str, default='relu', help="activation function")
        parent_parser.add_argument("--dropout", type=float, default=0.5, help="dropout probablity")
        parent_parser.add_argument("--ensemble", type=bool, default=True, help="whether to ensemble another MLP")
        parent_parser.add_argument("--ensemble_mlp_layer", type=int, nargs='+', default=[256,64], help="layer size of ensemble MLP")
        parent_parser.add_argument("--ensemble_activation", type=str, default='relu', help="activation function of ensemble MLP")
        parent_parser.add_argument("--ensemble_dropout", type=float, default=0.5, help="dropout probablity of ensemble MLP")
        return parent_parser
    
    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        num_fields = len(self.fields) - 1
        self.afn = nn.Sequential(
                    OrderedDict([
                        ("embeddings", 
                            ctr.Embeddings(
                                self.fields, 
                                self.embed_dim, 
                                train_data)),
                        ("logtransform_layer",
                            ctr.LogTransformLayer(
                                num_fields,
                                model_config['log_hidden_size'])),
                        ("mlp",
                            MLPModule(
                                [model_config['log_hidden_size'] * self.embed_dim] + model_config['mlp_layer'] + [1],
                                model_config['activation'], 
                                model_config['dropout'],
                                last_activation=False, 
                                last_bn=False))
                    ]))    
        if model_config['ensemble']:
            self.ensemble_embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
            self.ensemble_mlp = MLPModule(
                                    [num_fields * self.embed_dim] + model_config['ensemble_mlp_layer'] + [1],
                                    model_config['ensemble_activation'], 
                                    model_config['ensemble_dropout'],
                                    last_activation=False, 
                                    last_bn=False
                                )
            self.ensemble_fc = nn.Linear(2, 1)
            

    def score(self, batch):
        afn_score = self.afn(batch)
        if self.config['model']['ensemble']:
            ensemble_emb = self.ensemble_embedding(batch)
            ensemble_mlp_score = self.ensemble_mlp(
                                    ensemble_emb.view(ensemble_emb.size(0), -1)
                                )
            score = self.ensemble_fc(
                        torch.cat([afn_score, ensemble_mlp_score], dim=-1)
                    )
        else:
            score = afn_score
        score = score.squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
