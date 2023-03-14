from ast import Lambda
from typing import List, Dict
from recstudio.model.module.layers import CrossCompressUnit, FeatInterLayers, MLPModule
from recstudio.model import basemodel, loss_func, scorer, module
from recstudio.data import dataset
from recstudio.ann import sampler 
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch import optim

class MKR_kg_scorer(scorer.InnerProductScorer):
    def forward(self, query, items):
        output = super().forward(query, items)
        return torch.sigmoid(output) 

class MKRKGTower(basemodel.BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.L = config['model']['L']
        self.H = config['model']['H']
        self.kg_network_index = config['data']['kg_network_index']
        self.item_emb = None
        self.featInterLayers = None

    def _init_model(self, train_data):
        self.fhid = train_data.get_network_field(self.kg_network_index, 0, 0)
        self.frid = train_data.get_network_field(self.kg_network_index, 0, 1)
        self.ftid = train_data.get_network_field(self.kg_network_index, 0, 2)
        self.ent_emb = nn.Embedding(train_data.num_values(self.fhid), self.embed_dim, padding_idx=0)
        self.rel_emb = nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)  

        # high layers
        self.kge_mlp = []
        for _ in range(self.H):
            self.kge_mlp.append(self.embed_dim * 2)
        self.kge_mlp.append(self.embed_dim)
        self.kge_mlp = MLPModule(self.kge_mlp)
        super()._init_model(train_data)

    def _get_dataset_class(self):
        pass

    def _set_data_field(self, data):
        fhid = data.get_network_field(self.kg_network_index, 0, 0)
        frid = data.get_network_field(self.kg_network_index, 0, 1)
        ftid = data.get_network_field(self.kg_network_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, fhid, frid, ftid])

    def _get_sampler(self, train_data):
        return None

    def _get_score_func(self):
        return MKR_kg_scorer()

    def _get_loss_func(self):
        return None

    def _get_item_encoder(self, train_data):
        return module.VStackLayer(
            module.HStackLayer(
                self.item_emb, 
                self.ent_emb
            ),
            self.featInterLayers,
            module.LambdaLayer(lambda inputs: inputs[1])
        )

    def _get_query_encoder(self, train_data):
        relation_mlp = []
        for _ in range(self.L + 1):
            relation_mlp.append(self.embed_dim)
        relation_mlp = MLPModule(relation_mlp)
        return nn.Sequential(self.rel_emb, relation_mlp)

    def forward(self, batch_data, full_score):
        t_e = self.ent_emb(batch_data[self.ftid])
        h_e = self.item_encoder(batch_data[self.fhid]) # [batch_size, dim]
        r_e = self.query_encoder(batch_data[self.frid])
        head_relation_concat = torch.cat([h_e, r_e], dim=-1)
        tail_pred = self.kge_mlp(head_relation_concat)
        tail_pred = torch.sigmoid(tail_pred)

        pos_score = self.score_func(tail_pred, t_e)
        return {'score' : pos_score}
       
"""
MKR
    Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation(WWW'19)
    Reference:
        https://doi.org/10.1145/3308558.3313411
"""

class MKR(basemodel.BaseRetriever):
    """
    MKR is a deep end-to-end framework that utilizes knowledge graph embedding task to assist recommendation task. The two tasks are associated by cross&compress units, 
    which automatically share latent features and learn high-order interactions between items in recommender systems and entities in the knowledge graph.
    """
    def __init__(self, config):
        self.use_inner_product = config['model']['use_inner_product']
        self.L = config['model']['L']
        self.H = config['model']['H']
        self.dropout = config['model']['dropout']
        self.kge_interval = config['model']['kge_interval']
        self.kg_network_index = config['data']['kg_network_index']
        super().__init__(config)

    def _init_model(self, train_data):
        self.fhid = train_data.get_network_field(self.kg_network_index, 0, 0)
        self.item_emb = nn.Embedding(train_data.num_values(self.fhid), self.embed_dim, padding_idx=0)  
        self.num_items = train_data.num_items

        self.featInterLayers = FeatInterLayers(self.embed_dim, self.L, CrossCompressUnit)
        # KG
        self.kgTower = MKRKGTower(self.config)
        self.kgTower.item_emb = self.item_emb
        self.kgTower.featInterLayers = self.featInterLayers
        self.kgTower._init_model(train_data)
        super()._init_model(train_data)

    def _get_dataset_class():
        return dataset.TripletDataset
    
    def _set_data_field(self, data):
        fhid = data.get_network_field(self.kg_network_index, 0, 0)
        frid = data.get_network_field(self.kg_network_index, 0, 1)
        ftid = data.get_network_field(self.kg_network_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, fhid, frid, ftid])

    def current_epoch_trainloaders(self, nepoch) -> List:
        if (nepoch + 1) % self.kge_interval == 0:
            return self.trainloaders, False
        else:
            return self.trainloaders[0], False

    def _get_train_loaders(self, train_data : dataset.TripletDataset):
        rec_loader = train_data.train_loader(batch_size = self.config['train']['batch_size'], shuffle = True, 
                                            drop_last = False)
        kg_loader = train_data.network_feat[self.kg_network_index].loader(batch_size = self.config['train']['batch_size'], shuffle = True,
                                                                        drop_last = False)
        return [rec_loader, kg_loader]

    def _get_loss_func(self):
        return loss_func.BinaryCrossEntropyLoss()

    def _get_score_func(self):
        if self.use_inner_product == False:
            self.rs_mlp = []
            for _ in range(self.H):
                self.rs_mlp.append(self.embed_dim * 2)
            self.rs_mlp.append(1)
            self.rs_mlp = MLPModule(self.rs_mlp)
            return scorer.MLPScorer(self.rs_mlp)
        else:
            return scorer.InnerProductScorer()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items - 1)
        
    def _get_query_encoder(self, train_data):
        user_mlp = []
        for _ in range(self.L + 1):
            user_mlp.append(self.embed_dim)
        user_mlp = MLPModule(user_mlp)
        return nn.Sequential(
            nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0),
            user_mlp
        )

    def _get_item_encoder(self, train_data):
        return module.VStackLayer(
            module.HStackLayer(self.item_emb, self.kgTower.ent_emb),
            self.featInterLayers,
            module.LambdaLayer(lambda inputs: inputs[0])
        )
        
    def training_step(self, batch):
        if not (self.fhid in batch): 
            output = self.forward(batch, False)
            loss = self.loss_fn(None, **output['score'])
            return loss
        else:
            output = self.kgTower.forward(batch, False)
            loss = -output['score'].sum()
            return loss

    def _get_item_vector(self):
        index = torch.arange(1, self.num_items).to(self.device)
        return self.item_encoder(index)
