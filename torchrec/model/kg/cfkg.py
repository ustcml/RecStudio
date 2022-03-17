from torch.utils import data
from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
import torch

class CFKG(basemodel.TwoTowerRecommender):
    def init_model(self, train_data):
        super().init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        # kg  
        self.fhid = train_data.fhid
        self.ftid = train_data.ftid
        self.frid = train_data.frid
        self.rel_emb = torch.nn.Embedding(train_data.num_relations + 1, self.embed_dim, padding_idx=0)
        self.kg_sampler = sampler.UniformSampler(train_data.num_entities - 1, self.score_func)

        self.user_sampler = sampler.UniformSampler(train_data.num_users - 1, self.score_func)

    def get_dataset_class(self):
        return dataset.KnowledgeBasedDataset

    def set_train_loaders(self, train_data):
        return False

    def config_scorer(self):
        return scorer.NormScorer(p=2)

    def config_loss(self):
        return loss_func.HingeLoss(self.config['margin'])

    def build_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_entities, self.embed_dim, padding_idx=0)

    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def construct_query(self, batch_data):
        return self.user_encoder(batch_data[self.fuid]) + self.rel_emb.weight[-1]

    def _get_neg_emb(self, query, pos_id):
        pos_prob, neg_id, neg_prob = self.sampler(query, self.neg_count, pos_id)
        return pos_prob, self.item_encoder(neg_id), neg_prob
    
    def _get_neg_kg_emb(self, query, pos_id):
        pos_prob, neg_id, neg_prob = self.kg_sampler(query, self.neg_count, pos_id)
        return pos_prob, self.item_encoder(neg_id), neg_prob
    
    def _get_neg_user_emb(self, query, pos_id):
        pos_prob, neg_id, neg_prob = self.user_sampler(query, self.neg_count, pos_id)
        return pos_prob, self.user_encoder(neg_id), neg_prob

    def forward(self, batch_data, full_score):
        user_e = self.user_encoder(batch_data[self.fuid]) # [batch_size, dim]
        item_e = self.item_encoder(batch_data[self.fiid])
        inter_e = (self.rel_emb.weight[-1]).expand_as(user_e) # [dim]->[batch_size, dim]
        h_e = self.item_encoder(batch_data[self.fhid])
        t_e = self.item_encoder(batch_data[self.ftid])
        r_e = self.rel_emb(batch_data[self.frid])
        # get neg embeddings 
        pos_item_prob, neg_item_e, neg_item_prob = self._get_neg_emb(user_e, batch_data[self.fiid])
        pos_t_prob, neg_t_e, neg_t_prob = self._get_neg_kg_emb(h_e, batch_data[self.ftid])
        pos_user_prob, neg_user_e, neg_user_prob = self._get_neg_user_emb(item_e, batch_data[self.fuid])
        pos_h_prob, neg_h_e, neg_h_prob = self._get_neg_kg_emb(t_e, batch_data[self.fhid])
        # calculate rec_score and kg_score together
        # replace tail to get corrupt triple
        query = torch.cat([user_e, h_e]) + torch.cat([inter_e, r_e])
        pos = torch.cat([item_e, t_e])
        neg = torch.cat([neg_item_e, neg_t_e])
        tail_pos_score = self.score_func(query, pos) # [2 * batch_size]
        tail_neg_score = self.score_func(query, neg) # [2 * batch_size, neg_count]
        tail_score = (tail_pos_score, torch.cat([pos_item_prob, pos_t_prob]), tail_neg_score, torch.cat([neg_item_prob, neg_t_prob]))
        # replace head to get corrupt triple
        query = torch.cat([item_e, t_e])
        pos = torch.cat([user_e, h_e]) + torch.cat([inter_e, r_e])
        neg = torch.cat([neg_user_e, neg_h_e]) + torch.cat([inter_e, r_e]).unsqueeze(1)
        head_pos_score = self.score_func(query, pos)
        head_neg_score = self.score_func(query, neg)
        head_score = (head_pos_score, torch.cat([pos_user_prob, pos_h_prob]), head_neg_score, torch.cat([neg_user_prob, neg_h_prob]))
        return tail_score, head_score

    def training_step(self, batch, batch_idx):
        tail_score, head_score = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        return self.loss_fn(None, *tail_score) + self.loss_fn(None, *head_score)

    def get_item_vector(self):
        return self.item_encoder.weight[1 : self.num_items + 1]