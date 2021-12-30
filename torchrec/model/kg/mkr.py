from torchrec.model.kg.layers import MLPModule
from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
from torch import nn
import torch.nn.functional as F
import torch
import copy

class CrossCompressUnit(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight_vv = nn.Linear(self.embed_dim, 1, True)
        self.weight_ev = nn.Linear(self.embed_dim, 1, False)
        self.weight_ve = nn.Linear(self.embed_dim, 1, False)
        self.weight_ee = nn.Linear(self.embed_dim, 1, True)
    
    def forward(self, v, e):
        v = v.unsqueeze(-1) #[batch_size, dim, 1] or [batch_size, neg, dim, 1]
        e = e.unsqueeze(-2) #[batch_size, 1, dim] or [batch_size, neg, 1, dim]

        c_matrix = torch.matmul(v, e) # [batch_size, dim, dim]
        c_matrix_transpose = c_matrix.transpose(-1, -2)

        v_output = (self.weight_vv(c_matrix) + self.weight_ev(c_matrix_transpose)).squeeze(-1) # [batch_size, dim, 1] -> [batch_size, dim] 
        e_output = (self.weight_ve(c_matrix) + self.weight_ee(c_matrix_transpose)).squeeze(-1)

        return v_output, e_output

class MKR_item_encoder(nn.Module):
    def __init__(self, rec_item_emb, ent_emb, ccunit, L):
        super().__init__()
        self.rec_item_emb = rec_item_emb
        self.ent_emb = ent_emb
        self.ccunit = ccunit
        self.L = L
    
    def forward(self, items=None, entities=None):
        if items != None:
            v = self.rec_item_emb(items)
            e = self.ent_emb(items)
        else:
            v = self.rec_item_emb(entities)
            e = self.ent_emb(entities)
        for _ in range(self.L):
            v, e = self.ccunit(v, e)
        output = v if items != None else e 
        return output 

class Kg_scorer(scorer.InnerProductScorer):
    def forward(self, query, items):
        output = super().forward(query, items)
        return torch.sigmoid(output) 

# class Kg_loss_fn(loss_func.PairwiseLoss):
#     def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
#         pos_sum = pos_score.sum()
#         neg_sum = neg_score.sum()
#         return -(pos_sum - neg_sum)

class MKR(basemodel.TwoTowerRecommender):
    def __init__(self, config):
        self.use_inner_product = config['use_inner_product']
        self.L = config['L']
        self.H = config['H']
        self.dropout = config['dropout']
        self.kge_interval = config['kge_interval']
        super().__init__(config)

    def init_model(self, train_data):
        self.rec_item_emb = basemodel.Embedding(train_data.num_entities, self.embed_dim, padding_idx=0, bias=False)
        self.ent_emb = basemodel.Embedding(train_data.num_entities, self.embed_dim, padding_idx=0, bias=False)
        self.rel_emb = nn.Embedding(train_data.num_relations, self.embed_dim, padding_idx=0)
        self.ccunit = CrossCompressUnit(self.embed_dim)     
        super().init_model(train_data)
        self.num_items = train_data.num_items
        # kg id
        self.fhid = train_data.fhid
        self.ftid = train_data.ftid
        self.frid = train_data.frid
        # low layers 
        self.tail_mlp = []
        self.user_mlp = []
        for _ in range(self.L):
            self.user_mlp.extend([self.embed_dim, self.embed_dim])
            self.tail_mlp.extend([self.embed_dim, self.embed_dim])
        self.tail_mlp = MLPModule(self.tail_mlp)
        self.user_mlp = MLPModule(self.user_mlp)
        # high layers
        self.kge_mlp = []
        for _ in range(self.H - 1):
            self.kge_mlp.extend([self.embed_dim * 2, self.embed_dim * 2])
        self.kge_mlp.extend([self.embed_dim * 2, self.embed_dim])
        self.kge_mlp = MLPModule(self.kge_mlp)
        # kge
        self.kg_score_func = Kg_scorer()
        # self.kg_loss_fn = Kg_loss_fn()
        self.kg_sampler = sampler.UniformSampler(train_data.num_entities - 1)

    def get_dataset_class(self):
        return dataset.KnowledgeBasedDataset

    def set_train_loaders(self, train_data):
        kg_train_data = copy.copy(train_data)
        kg_train_data.kg_state = 1
        train_data.loaders = [train_data.loader, kg_train_data.loader]
        train_data.nepoch = [self.kge_interval, 1]
        return False
          
    def config_scorer(self):
        if self.use_inner_product == False:
            self.rs_mlp = []
            for _ in range(self.H - 1):
                self.rs_mlp.extend([self.embed_dim * 2, self.embed_dim * 2])
            self.rs_mlp = MLPModule(self.rs_mlp)
            self.rs_mlp.add_modules(nn.Dropout(self.dropout), nn.Linear(self.embed_dim * 2, 1))
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
        return MKR_item_encoder(self.rec_item_emb, self.ent_emb, self.ccunit, self.L)

    def kg_forward(self, batch_data, batch_idx):
        t_e = self.ent_emb(batch_data[self.ftid])
        r_e = self.rel_emb(batch_data[self.frid])

        h_e = self.item_encoder(entities=batch_data[self.fhid]) # [batch_size, dim]
        r_e = self.tail_mlp(r_e) # [batch_size, dim]
        head_relation_concat = torch.cat([h_e, r_e], dim=-1)
        tail_pred = self.kge_mlp(head_relation_concat)
        tail_pred = torch.sigmoid(tail_pred)

        pos_score = self.kg_score_func(tail_pred, t_e)
        return pos_score
        
    def training_step(self, batch, batch_idx):
        if self.current_epoch % (self.kge_interval + 1) < self.kge_interval:
            y_h = self.forward(batch, batch_idx)
            return self.loss_fn(None, *y_h)
        else:
            y_h = self.kg_forward(batch, batch_idx)
            return -y_h.sum()

    def get_item_vector(self):
        index = torch.arange(1, self.num_items).to(self.device)
        return self.item_encoder(index)

    # def on_train_epoch_start(self):
    #     super().on_train_epoch_start()
    #     if (self.current_epoch + 1) % 3 == 0:
    #         self.val_check = False
    #     else:
    #         self.val_check = True

    # def validation_step(self, batch, batch_idx):
    #     if (self.current_epoch + 1) % 3 != 0:
    #         return super().validation_step(batch, batch_idx)

    # def validation_epoch_end(self, outputs):
    #     if (self.current_epoch + 1) % 3 != 0:
    #         super().validation_epoch_end(outputs)

    # def training_epoch_end(self, outputs):
    #     if (self.current_epoch + 1) % 3 != 0:
    #         super().training_epoch_end(outputs)
    #     # else:





# TODO tow optimizers 
# TODO use train loop
# TODO Embloss. l2loss in paper code is different from weight_decay.