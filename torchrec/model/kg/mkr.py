from torchrec.model.kg.loops import MKRFitLoop
from torchrec.model.kg.layers import MLPModule
from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
from torch import optim

class CrossCompressUnit(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight_vv = nn.Linear(self.embed_dim, 1, False)
        self.weight_ev = nn.Linear(self.embed_dim, 1, False)
        self.weight_ve = nn.Linear(self.embed_dim, 1, False)
        self.weight_ee = nn.Linear(self.embed_dim, 1, False)
        self.bias_v = Parameter(data=torch.zeros(self.embed_dim), requires_grad=True)
        self.bias_e = Parameter(data=torch.zeros(self.embed_dim), requires_grad=True)
    
    def forward(self, inputs):
        v = inputs[0].unsqueeze(-1) #[batch_size, dim, 1] or [batch_size, neg, dim, 1]
        e = inputs[1].unsqueeze(-2) #[batch_size, 1, dim] or [batch_size, neg, 1, dim]

        c_matrix = torch.matmul(v, e) # [batch_size, dim, dim] or [batch_size, neg, dim, dim]
        c_matrix_transpose = c_matrix.transpose(-1, -2)

        v_output = (self.weight_vv(c_matrix) + self.weight_ev(c_matrix_transpose)).squeeze(-1) # [batch_size, dim, 1] -> [batch_size, dim] 
        v_output = v_output + self.bias_v
        e_output = (self.weight_ve(c_matrix) + self.weight_ee(c_matrix_transpose)).squeeze(-1)
        e_output = e_output + self.bias_e 

        return (v_output, e_output)

class MKR_item_encoder(nn.Module):
    def __init__(self, rec_item_emb, ent_emb, embed_dim, L):
        super().__init__()
        self.rec_item_emb = rec_item_emb
        self.ent_emb = ent_emb
        self.embed_dim = embed_dim
        self.L = L
        self.cc_units = nn.Sequential()
        for id in range(self.L):
            self.cc_units.add_module(f'cc_unit[{id}]', CrossCompressUnit(self.embed_dim))
        
    def forward(self, items=None, entities=None):
        if items != None:
            v = self.rec_item_emb(items)
            e = self.ent_emb(items)
        else:
            v = self.rec_item_emb(entities)
            e = self.ent_emb(entities)
        v, e = self.cc_units((v, e))
        output = v if items != None else e 
        return output 

class Kg_scorer(scorer.InnerProductScorer):
    def forward(self, query, items):
        output = super().forward(query, items)
        return torch.sigmoid(output) 

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
        super().init_model(train_data)
        self.num_items = train_data.num_items
        # kg id
        self.fhid = train_data.fhid
        self.ftid = train_data.ftid
        self.frid = train_data.frid
        # low layers 
        self.relation_mlp = []
        self.user_mlp = []
        for _ in range(self.L + 1):
            self.user_mlp.append(self.embed_dim)
            self.relation_mlp.append(self.embed_dim)
        self.relation_mlp = MLPModule(self.relation_mlp)
        self.user_mlp = MLPModule(self.user_mlp)
        # high layers
        self.kge_mlp = []
        for _ in range(self.H):
            self.kge_mlp.append(self.embed_dim * 2)
        self.kge_mlp.append(self.embed_dim)
        self.kge_mlp = MLPModule(self.kge_mlp)
        # kge
        self.kg_score_func = Kg_scorer()
        # self.kg_loss_fn = Kg_loss_fn()
        self.kg_sampler = sampler.UniformSampler(train_data.num_entities - 1)

    def get_dataset_class(self):
        return dataset.KnowledgeBasedDataset

    def set_train_loaders(self, train_data):
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

    def configure_optimizers(self):
        opt_rec = self.get_optimizer(self.parameters())
        opt_kg = optim.Adam(self.parameters(), lr=self.config['learning_rate_kg'], weight_decay=self.config['weight_decay'])
        return [opt_rec, opt_kg]

    def build_user_encoder(self, train_data):
        return nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def construct_query(self, batch_data):
        return self.user_mlp(self.user_encoder(batch_data[self.fuid]))

    def build_item_encoder(self, train_data):
        return MKR_item_encoder(self.rec_item_emb, self.ent_emb, self.embed_dim, self.L)

    def kg_forward(self, batch_data, batch_idx):
        t_e = self.ent_emb(batch_data[self.ftid])
        r_e = self.rel_emb(batch_data[self.frid])

        h_e = self.item_encoder(entities=batch_data[self.fhid]) # [batch_size, dim]
        r_e = self.relation_mlp(r_e) # [batch_size, dim]
        head_relation_concat = torch.cat([h_e, r_e], dim=-1)
        tail_pred = self.kge_mlp(head_relation_concat)
        tail_pred = torch.sigmoid(tail_pred)

        pos_score = self.kg_score_func(tail_pred, t_e)
        return pos_score
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        if not (self.fhid in batch) and optimizer_idx == 0:
            y_h = self.forward(batch, batch_idx)
            loss = self.loss_fn(None, *y_h)
            return {'loss' : loss, 'loss_rec' : loss.detach()}
        elif self.fhid in batch and optimizer_idx == 1:
            y_h = self.kg_forward(batch, batch_idx)
            loss = -y_h.sum()
            return {'loss' : loss, 'loss_kg' : loss.detach()}

    def training_epoch_end(self, outputs):
        for output in outputs:
            output.pop('loss')
        return super().training_epoch_end(outputs)
   
    def get_item_vector(self):
        index = torch.arange(1, self.num_items).to(self.device)
        return self.item_encoder(index)


# TODO Embloss. l2loss in paper code is different from weight_decay.
# TODO in kg_forward, there is no negative sampling. 