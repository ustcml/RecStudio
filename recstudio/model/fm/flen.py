import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule, get_act

r"""
FLEN
######################

Paper Reference:
    FLEN: Leveraging Field for Scalable CTR Prediction (DLP KDD'20)
    https://dlp-kdd.github.io/dlp-kdd2020/assets/pdf/a3-chen.pdf
"""

class FLEN(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        if model_config.get('fields', None) is None:
            fields = [f.fields for f in train_data._get_feat_list()]  
        else:
            fields = model_config['fields']
            all_fields = set()
            for f in fields:
                if all_fields.intersection(set(f)) not in [{self.fuid}, {self.fiid}, 
                                                        {self.fuid, self.fiid}, set()] :
                    raise ValueError('Expect no intersection between fields '
                                        f'expcept {self.fuid} and {self.fiid}, '
                                        f'but got mutilple {all_fields.intersection(set(f))}.')
                all_fields = all_fields.union(set(f))
            if len(all_fields) != self.embedding.num_features:
                raise ValueError(f'Expect fields consist {self.embedding.num_features}, '
                                f'but got {all_fields - self.fields}.')
                
        self.fwbi = ctr.FieldWiseBiInteraction(
                        self.embed_dim,
                        train_data,
                        model_config['activation'],
                        model_config['dropout'],
                        fields)
        self.mlp = MLPModule(
                        [self.embedding.num_features*self.embed_dim] + model_config['mlp_layer'],
                        model_config['activation'], 
                        model_config['dropout'],
                        batch_norm=True,
                        last_activation=True, 
                        last_bn=True)
        self.fc = nn.Linear(model_config['mlp_layer'][-1] + self.embed_dim + 1, 1, bias=False)
            
    def score(self, batch):
        emb = self.embedding(batch)
        field_embs = []
        for field in self.fwbi.fields:
            field_idx = [list(self.embedding.embeddings).index(f) for f in field if f != self.frating]
            field_embs.append(emb[:, field_idx, :])
        fwbi_out = self.fwbi(batch, field_embs)
        mlp_out = self.mlp(emb.flatten(1))
        score = self.fc(torch.cat([mlp_out, fwbi_out], dim=-1)).squeeze(-1)
        return {'score': score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
