import torch
import torch.nn as nn
from collections import defaultdict
from recstudio.model.multitask.hardshare import HardShare
from ..module import ctr, MLPModule, AttentionLayer

r"""
AITM
######################

Paper Reference:
    Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising (KDD'21)
    https://doi.org/10.1145/3447548.3467071
"""

class AITM(HardShare):

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        assert isinstance(self.frating, list), f'Expect rating_field to be a list, but got {self.frating}.'
        self.towers = nn.ModuleDict({
                            r: MLPModule(
                                [self.embedding.num_features*self.embed_dim] + model_config['tower_mlp_layer'],
                                model_config['tower_activation'], 
                                model_config['tower_dropout'],
                                batch_norm=model_config['tower_batch_norm'])
                            for r in self.frating
                        })
        self.att_layers = nn.ModuleDict({
                            r: AttentionLayer(
                                model_config['tower_mlp_layer'][-1],
                                attention_type='multi-head',
                                n_head=1)
                            for r in self.frating[1:]
                        })
        self.info_layers = nn.ModuleDict({
                                r: nn.Sequential(
                                        nn.Linear(model_config['tower_mlp_layer'][-1], model_config['tower_mlp_layer'][-1]),
                                        nn.ReLU())
                                for r in self.frating[:-1]
                            })
        self.fc_layers = nn.ModuleDict({
                            r: nn.Linear(model_config['tower_mlp_layer'][-1], 1)
                            for r in self.frating
                        })
            
    def score(self, batch):
        emb = self.embedding(batch).flatten(1)
        score = defaultdict(dict)
        for r, tower in self.towers.items():
            tower_out = tower(emb)                                          # B x T
            if r not in self.att_layers:
                ait_out = tower_out                                         # B x T
            else:
                u = torch.stack([info_out, tower_out], dim=1)               # B x 2 x T
                ait_out = self.att_layers[r](u, u, u).sum(1)                # B x T
            score[r]['score'] = self.fc_layers[r](ait_out).squeeze(-1)
            if r in self.info_layers:
                info_out = self.info_layers[r](ait_out)                     # B x T
        return score
    
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
        calibrator_loss = sum((torch.max(
                                    y_h[r_]['pos_score'] - y_h[r]['pos_score'], 
                                    torch.tensor(0., device=self.device))).mean() 
                                for r_, r in zip(self.frating[1:], self.frating[:-1]))
        loss['loss'] += calibrator_loss
        return loss
