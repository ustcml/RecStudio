import torch
import torch.nn as nn
from collections import defaultdict
from recstudio.model.multitask.hardshare import HardShare
from ..module import ctr, MLPModule

r"""
PLE
######################

Paper Reference:
    Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (RecSys'20)
    https://dl.acm.org/doi/10.1145/3383313.3412236
"""

class PLE(HardShare):

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        assert isinstance(self.frating, list), f'Expect rating_field to be a list, but got {self.frating}.'
        self.extraction_layers = nn.Sequential(*[
                                    ctr.ExtractionLayer(
                                        self.embedding.num_features * self.embed_dim if i == 0 else model_config['expert_mlp_layer'][-1],
                                        model_config['specific_experts_per_task'],
                                        len(self.frating),
                                        model_config['num_shared_experts'],
                                        True if i != model_config['num_levels'] - 1 else False,
                                        model_config['expert_mlp_layer'],
                                        model_config['expert_activation'],
                                        model_config['expert_dropout'],
                                        model_config['gate_mlp_layer'],
                                        model_config['gate_activation'],
                                        model_config['gate_dropout'])
                                    for i in range(model_config['num_levels'])
                                ])
        self.towers = nn.ModuleDict({
                            r: MLPModule(
                                [model_config['expert_mlp_layer'][-1]] + model_config['tower_mlp_layer'] + [1],
                                model_config['tower_activation'], 
                                model_config['tower_dropout'],
                                batch_norm=model_config['tower_batch_norm'],
                                last_activation=False, 
                                last_bn=False)
                            for r in self.frating
                        })
            
    def score(self, batch):
        emb = self.embedding(batch).flatten(1)
        extraction_out = self.extraction_layers([emb] * (len(self.frating) + 1))
        score = defaultdict(dict)
        for i, (r, tower) in enumerate(self.towers.items()):
            score[r]['score'] = tower(extraction_out[i]).squeeze(-1)
        return score