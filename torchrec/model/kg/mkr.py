from torchrec.model.kg.loops import MKRFitLoop
from torchrec.model.module.layers import CrossCompressUnit, FeatInterLayers, MLPModule
from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch import optim

class MKR_kg_scorer(scorer.InnerProductScorer):
    def forward(self, query, items):
        output = super().forward(query, items)
        return torch.sigmoid(output) 

class MKR_entity_encoder(nn.Module):
    def __init__(self, item_emb, ent_emb, featInterLayers):
        super().__init__()
        self.item_emb = item_emb
        self.ent_emb = ent_emb
        self.featInterLayers = featInterLayers
        
    def forward(self, entities):
        v = self.item_emb(entities)
        e = self.ent_emb(entities)
        v, e = self.featInterLayers(v, e)
        return e

class MKRKGTower(basemodel.ItemTowerRecommender):
    def __init__(self, config):
        self.L = config['L']
        self.H = config['H']
        self.kg_index = config['kg_network_index']
        self.item_emb = None
        self.featInterLayers = None
        super().__init__(config)

    def init_model(self, train_data):
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 1)
        self.frid = train_data.get_network_field(self.kg_index, 0, 2)
        self.ent_emb = basemodel.Embedding(train_data.num_values(self.fhid), self.embed_dim, padding_idx=0, bias=False)
        self.rel_emb = nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)  
        # low layers
        self.relation_mlp = []
        for _ in range(self.L + 1):
            self.relation_mlp.append(self.embed_dim)
        self.relation_mlp = MLPModule(self.relation_mlp)
        # high layers
        self.kge_mlp = []
        for _ in range(self.H):
            self.kge_mlp.append(self.embed_dim * 2)
        self.kge_mlp.append(self.embed_dim)
        self.kge_mlp = MLPModule(self.kge_mlp)
        super().init_model(train_data)

    def get_dataset_class(self):
        pass

    def build_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_values(self.fhid) - 1)

    def config_scorer(self):
        return MKR_kg_scorer()

    def config_loss(self):
        return None

    def build_item_encoder(self, train_data):
        return MKR_entity_encoder(self.item_emb, self.ent_emb, self.featInterLayers)

    def construct_query(self, batch_data):
        r_e = self.rel_emb(batch_data[self.frid])
        r_e = self.relation_mlp(r_e) # [batch_size, dim]
        return r_e

    def forward(self, batch_data):
        t_e = self.ent_emb(batch_data[self.ftid])
        h_e = self.item_encoder(batch_data[self.fhid]) # [batch_size, dim]
        r_e = self.construct_query(batch_data)
        head_relation_concat = torch.cat([h_e, r_e], dim=-1)
        tail_pred = self.kge_mlp(head_relation_concat)
        tail_pred = torch.sigmoid(tail_pred)

        pos_score = self.score_func(tail_pred, t_e)
        return pos_score

class MKR_item_encoder(nn.Module):
    def __init__(self, item_emb, ent_emb, featInterLayers):
        super().__init__()
        self.item_emb = item_emb
        self.ent_emb = ent_emb
        self.featInterLayers = featInterLayers
        
    def forward(self, items):
        v = self.item_emb(items)
        e = self.ent_emb(items)
        v, e = self.featInterLayers(v, e)
        return v

"""
MKR
    Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation(WWW'19)
    Reference:
        https://doi.org/10.1145/3308558.3313411
"""

class MKR(basemodel.TwoTowerRecommender):
    """
    MKR is a deep end-to-end framework that utilizes knowledge graph embedding task to assist recommendation task. The two tasks are associated by cross&compress units, 
    which automatically share latent features and learn high-order interactions between items in recommender systems and entities in the knowledge graph.
    """
    def __init__(self, config):
        self.use_inner_product = config['use_inner_product']
        self.L = config['L']
        self.H = config['H']
        self.dropout = config['dropout']
        self.kge_interval = config['kge_interval']
        self.kg_index = config['kg_network_index']
        super().__init__(config)

    def init_model(self, train_data):
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.item_emb = basemodel.Embedding(train_data.num_values(self.fhid), self.embed_dim, padding_idx=0, bias=False)  
        self.num_items = train_data.num_items
        # low layers 
        self.user_mlp = []
        for _ in range(self.L + 1):
            self.user_mlp.append(self.embed_dim)
        self.user_mlp = MLPModule(self.user_mlp)
        self.featInterLayers = FeatInterLayers(self.embed_dim, self.L, CrossCompressUnit)
        #KG
        self.kgTower = MKRKGTower(self.config)
        self.kgTower.item_emb = self.item_emb
        self.kgTower.featInterLayers = self.featInterLayers
        self.kgTower.init_model(train_data)
        
        super().init_model(train_data)

    def get_dataset_class(self):
        return dataset.MFDataset

    def set_train_loaders(self, train_data):
        train_data.loaders = [train_data.loader, train_data.network_feat[self.kg_index].loader]
        train_data.nepoch = [self.kge_interval, 1]
        return False
          
    def config_fitloop(self, trainer):
        trainer.fit_loop = MKRFitLoop(self.kge_interval)

    def config_scorer(self):
        if self.use_inner_product == False:
            self.rs_mlp = []
            for _ in range(self.H):
                self.rs_mlp.append(self.embed_dim * 2)
            self.rs_mlp.append(1)
            self.rs_mlp = MLPModule(self.rs_mlp)
            return scorer.MLPScorer(self.rs_mlp)
        else:
            return scorer.InnerProductScorer()
            
    def config_loss(self):
        return loss_func.BinaryCrossEntropyLoss()

    def build_user_encoder(self, train_data):
        return nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def construct_query(self, batch_data):
        return self.user_mlp(self.user_encoder(batch_data[self.fuid]))

    def build_item_encoder(self, train_data):
        return MKR_item_encoder(self.item_emb, self.kgTower.ent_emb, self.featInterLayers)
        
    def training_step(self, batch, batch_idx):
        if not (self.fhid in batch):
            y_h = self.forward(batch, batch_idx)
            loss = self.loss_fn(None, *y_h)
            return {'loss' : loss, 'loss_rec' : loss.detach()}
        else:
            y_h = self.kgTower.forward(batch)
            loss = -y_h.sum()
            return {'loss' : loss, 'loss_kg' : loss.detach()}

    def training_epoch_end(self, outputs):
        for output in outputs:
            output.pop('loss')
        return super().training_epoch_end(outputs)
   
    def get_item_vector(self):
        index = torch.arange(1, self.num_items).to(self.device)
        return self.item_encoder(index)
