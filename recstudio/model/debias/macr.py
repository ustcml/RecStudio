import torch
from .debiasedretriever import DebiasedRetriever
from ..loss_func import BCELoss
from recstudio.model import basemodel, scorer
from recstudio.model.module import MLPModule

r"""
MACR
######

Paper Reference:
    Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System (KDD'21)
    https://doi.org/10.1145/3447548.3467289
"""

class MACR(DebiasedRetriever):  
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('MACR')
        parent_parser.add_argument("--c", type=float, default=0.0, help='reference status')
        parent_parser.add_argument("--alpha", type=float, default=1e-3, help='weight of user loss')
        parent_parser.add_argument("--beta", type=float, default=1e-3, help='weight of item loss')
        return parent_parser        
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.backbone['matching'].loss_fn = None
        self.backbone['matching'].sampler = None

    def _get_score_func(self):
        # For only topk() function
        class MACRScorer(scorer.InnerProductScorer):
            def __init__(self, c, user_module, item_module):
                super().__init__()
                self.c = c
                self.user_module = user_module  # shared
                self.item_module = item_module  # shared
            def forward(self, query, items):
                yk = super().forward(query, items)
                yu = self.user_module(query).squeeze()
                yi = self.item_module(items).squeeze()
                yui = (yk - self.c) * torch.outer(yu, yi)
                return yui
        
        assert self.config['user_module']['mlp_layers'][0] == self.config['model']['embed_dim']
        assert self.config['item_module']['mlp_layers'][0] == self.config['model']['embed_dim']
        assert self.config['user_module']['mlp_layers'][-1] == 1
        assert self.config['item_module']['mlp_layers'][-1] == 1
        self.user_module = MLPModule(**self.config['user_module'])
        self.item_module = MLPModule(**self.config['item_module'])
        if self.config['user_module']['activation_func'].lower() != 'sigmoid':
            self.user_module.add_modules(torch.nn.Sigmoid())
        if self.config['item_module']['activation_func'].lower() != 'sigmoid':
            self.item_module.add_modules(torch.nn.Sigmoid())
        return MACRScorer(self.config['train']['c'], self.user_module, self.item_module)
         
    def _get_loss_func(self):
        return BCELoss()

    def _get_final_loss(self, loss: dict, output: dict, batch : dict):
        label = batch[self.frating]
        score_u = self.user_module(output['matching']['query']).squeeze()
        score_i = self.item_module(output['matching']['item']).squeeze()
        score_click = torch.sigmoid(output['matching']['score']['pos_score'] * score_u * score_i)
        loss_click = self.loss_fn(label=label, pos_score=score_click)
        loss_u = self.loss_fn(label=label, pos_score=score_u)
        loss_i = self.loss_fn(label=label, pos_score=score_i)
        return loss_click + self.config['train']['alpha'] * loss_u + self.config['train']['beta'] * loss_i