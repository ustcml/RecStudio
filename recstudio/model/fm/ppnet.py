import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
PPNet
######################

    Used in Kuai 2019
"""

class BCEWithLogitLossWithAux(loss_func.BCEWithLogitLoss):
    def forward(self, aux_score, label, pos_score):
        return super().forward(label, aux_score) + super().forward(label, pos_score)

class PPNet(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        self.mlp = MLPModule([self.embedding.num_features*self.embed_dim] + model_config['mlp_layer'] + [1],
                        model_config['activation'], 
                        model_config['dropout'],
                        last_activation=False, 
                        last_bn=False
                    )
        if model_config['id_fields'] is None:
            id_fields = []
            if self.fuid is not None:
                id_fields.append(self.fuid)
            if self.fiid is not None:
                id_fields.append(self.fiid)
            if len(id_fields) == 0:
                raise ValueError('Expect id_fields, but got None.')
        else:
            id_fields = model_config['id_fields']
        self.id_embedding = ctr.Embeddings(id_fields, model_config['id_embed_dim'], train_data)
        pp_hidden_dims = [self.embedding.num_features*self.embed_dim] + model_config['pp_hidden_dims']
        self.ppnet = nn.ModuleList([
                        ctr.PPLayer(
                            pp_hidden_dims[i : i + 2],
                            self.embedding.num_features*self.embed_dim + len(id_fields)*model_config['id_embed_dim'],
                            model_config['gate_hidden_dims'][i],
                            model_config['activation'],
                            model_config['dropout'],
                            model_config['batch_norm']) 
                        for i in range(len(pp_hidden_dims) - 1)
                    ])
        self.fc = nn.Linear(pp_hidden_dims[-1], 1)

    def score(self, batch):
        emb = self.embedding(batch)
        mlp_score = self.mlp(emb.flatten(1)).squeeze(-1)
        
        id_emb = self.id_embedding(batch)
        gate_in = torch.cat([emb.flatten(1).detach(), id_emb.flatten(1)], dim=-1)
        mlp_in = emb.flatten(1).detach()
        for pplayer in self.ppnet:
            mlp_in = pplayer(gate_in, mlp_in)
        ppnet_score = self.fc(mlp_in).squeeze(-1)
        return {'aux_score' : mlp_score, 'score': ppnet_score}

    def _get_loss_func(self):
        return BCEWithLogitLossWithAux()
    
    def training_step(self, batch):
        y_h, output = self.forward(batch)
        loss = self.loss_fn(output['aux_score'], **y_h)
        return loss
