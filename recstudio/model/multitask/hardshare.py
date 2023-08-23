import torch
import torch.nn as nn
from collections import defaultdict
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule

r"""
HardShare
######################

Paper Reference:
    An overview of multi-task learning in deep neural networks ('17)
    https://arxiv.org/abs/1706.05098
"""

class HardShare(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        self.bottom_mlp = MLPModule(
                            [self.embedding.num_features * self.embed_dim] + model_config['bottom_mlp_layer'],
                            model_config['bottom_activation'], 
                            model_config['bottom_dropout'],
                            batch_norm=model_config['bottom_batch_norm'])
        assert isinstance(self.frating, list), f'Expect rating_field to be a list, but got {self.frating}.'
        self.top_mlp = nn.ModuleDict({
                            r: MLPModule(
                                [model_config['bottom_mlp_layer'][-1]] + model_config['top_mlp_layer'] + [1],
                                model_config['top_activation'], 
                                model_config['top_dropout'],
                                last_activation=False, 
                                batch_norm=model_config['top_batch_norm'])
                            for r in self.frating
                        })
            
    def score(self, batch):
        emb = self.embedding(batch)
        shared_emb = self.bottom_mlp(emb.flatten(1))
        score = defaultdict(dict)
        for r, top_mlp in self.top_mlp.items():
            score[r]['score'] = top_mlp(shared_emb).squeeze(-1)
        return score

    def _get_loss_func(self):
        return BCEWithLogitLoss()
    
    def training_step(self, batch):
        y_h, _ = self.forward(batch)
        loss = {}
        for r in self.frating:
            loss[r] = self.loss_fn(**y_h[r])
        
        weights = self.config['train'].get('weights', [1.0]*len(self.frating))
        if weights is None:
            weights = [1.0]*len(self.frating)
        assert len(weights) == len(self.frating), \
            f'Expect {len(self.frating)} float(s) for weights, but got {self.config["train"]["weights"]}.'
        weights = torch.tensor(weights, device=self.device).softmax(0)
        
        loss['loss'] = sum(w*v for w, (_, v) in zip(weights, loss.items()))
        return loss
