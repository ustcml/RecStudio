import torch.nn as nn
from collections import OrderedDict
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, MLPModule
from recstudio.data.dataset import TripletDataset
from ..loss_func import BCEWithLogitLoss


class NFM(BaseRanker):
    
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("NFM")
        parent_parser.add_argument("--mlp_layer", type=int, nargs='+', default=[128,128,128], help="the MLP layer size")
        parent_parser.add_argument("--activation", type=str, default='sigmoid', help="activation function")
        parent_parser.add_argument("--dropout", type=float, default=0.3, help="dropout probablity")
        parent_parser.add_argument("--batch_norm", type=bool, default=False, help="whether to use batch_norm")
        return parent_parser

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        model_config = self.config['model']
        self.nfm = nn.Sequential(
                        OrderedDict([
                            ("embedding",
                                ctr.Embeddings(
                                    self.fields, 
                                    self.embed_dim, 
                                    train_data)),
                            ("fm_layer",
                                ctr.FMLayer()),
                            ("batch_norm",
                                nn.BatchNorm1d(self.embed_dim)),
                            ("mlp",
                                MLPModule(
                                    [self.embed_dim]+model_config['mlp_layer']+[1],
                                    model_config['activation'],
                                    model_config['dropout'],
                                    batch_norm=model_config['batch_norm'],
                                    last_activation=False, last_bn=False))
                        ]))

    def score(self, batch):
        linear_score = self.linear(batch)
        mlp_score = self.nfm(batch).squeeze(-1)
        return {'score' : linear_score + mlp_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
