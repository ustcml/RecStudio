import torch.nn as nn
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule, HStackLayer, VStackLayer

r"""
FinalMLP
######################

Paper Reference:
    FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction (AAAI'23)
    https://arxiv.org/abs/2304.00902
"""

class FinalMLP(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        num_fields = self.embedding.num_features
        stream1_fields = train_data.user_feat.fields if model_config.get('fields1', None) is None else model_config['fields1']
        stream2_fields = train_data.item_feat.fields if model_config.get('fields2', None) is None else model_config['fields2']
        if set(stream1_fields + stream2_fields) - set(list(self.embedding.field2types)) != set():
            raise ValueError(f'Expect the fields of two streams are contained in {self.embedding.field2types}, '
                             f'but got {set(stream1_fields + stream2_fields) - set(list(self.embedding.field2types))}.')
            
        if model_config['feature_selection']:
            self.fs = ctr.FeatureSelection(
                            stream1_fields,
                            stream2_fields,
                            self.embed_dim,
                            num_fields,
                            train_data,
                            model_config['fs_mlp_layer']
                        )
        self.towers = VStackLayer(OrderedDict([
                        ('mlps',
                            HStackLayer(
                                MLPModule(
                                    [num_fields*self.embed_dim] + model_config['mlp_layer1'],
                                    model_config['activation1'], 
                                    model_config['dropout1'],
                                    batch_norm=model_config['batch_norm1']),
                                MLPModule(
                                    [num_fields*self.embed_dim] + model_config['mlp_layer2'],
                                    model_config['activation2'], 
                                    model_config['dropout2'],
                                    batch_norm=model_config['batch_norm2']))),
                        ('fusion',
                            ctr.MultiHeadBilinearFusion(
                                model_config['n_head'],
                                model_config['mlp_layer1'][-1],
                                model_config['mlp_layer2'][-1],
                                output_dim=1))
                    ]))
        
    def score(self, batch):
        emb = self.embedding(batch)
        if self.config['model']['feature_selection']:
            emb1, emb2 = self.fs(batch, emb.flatten(1))
        else:
            emb1 = emb2 = emb.flatten(1)
        score = self.towers((emb1, emb2)).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
