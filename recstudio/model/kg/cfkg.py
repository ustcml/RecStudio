from torch.utils import data
from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler 
import torch
"""
CFKG
#################
    Learning over Knowledge-Base Embeddings for Recommendation(SIGIR'18)
    Reference: 
        https://arxiv.org/abs/1803.06540
"""

class CFKGQueryEncoder(torch.nn.Module):

    def __init__(self, train_data, embed_dim, rel_emb, fuid) -> None:
        super().__init__()
        self.fuid = fuid
        self.rel_emb = rel_emb
        self.user_emb = torch.nn.Embedding(train_data.num_users, embed_dim, padding_idx=0)
    
    def forward(self, batch_users, add_inter_emb=True):
        if add_inter_emb: 
            return self.user_emb(batch_users) + self.rel_emb.weight[-1]
        else:
            return self.user_emb(batch_users)
        

class CFKG(basemodel.BaseRetriever):
    """
    In CFKG, user-item interaction is incorporated into the knowledge graph as triplets i.e. (user, interact, item) to form user-item knowledge graph.
    And then, CFKG conducts collaborative fltering on this graph to provide personalized recommendations using TransE to learn the representation of users and items.
    In this implementation, we sample users, items and entities from the users and items in recommender data and the entities in knowledge graph respectively, while
    in the original paper, they are all sampled from user-item knowledge graph.
    """
    def __init__(self, config):
        self.kg_index = config['data']['kg_network_index']
        super().__init__(config)

    def _init_model(self, train_data):
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        # kg  
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 1)
        self.frid = train_data.get_network_field(self.kg_index, 0, 2)
        self.rel_emb = torch.nn.Embedding(train_data.num_values(self.frid) + 1, self.embed_dim, padding_idx=0)
        self.kg_sampler = sampler.UniformSampler(train_data.num_values(self.fhid) - 1, self.score_func)
        self.user_sampler = sampler.UniformSampler(train_data.num_users - 1, self.score_func)
        super()._init_model(train_data)


    def _get_dataset_class():
        return dataset.TripletDataset
    
    def _set_data_field(self, data : dataset.TripletDataset):
        fhid = data.get_network_field(self.kg_index, 0, 0)
        ftid = data.get_network_field(self.kg_index, 0, 1)
        frid = data.get_network_field(self.kg_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, fhid, frid, ftid])

    def _get_train_loaders(self, train_data: dataset.TripletDataset):
        rec_loader = train_data.train_loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        kg_loader = train_data.network_feat[self.kg_index].loader(batch_size = self.config['train']['batch_size'], shuffle = True, drop_last = False)
        return [rec_loader, kg_loader]
    
    def current_epoch_trainloaders(self, nepoch):
        return self.trainloaders, True 

    def _get_score_func(self):
        return scorer.NormScorer(p=2)

    def _get_loss_func(self):
        return loss_func.HingeLoss(self.config['model']['margin'])

    def _get_item_encoder(self, train_data : dataset.TripletDataset):
        return torch.nn.Embedding(train_data.num_values(train_data.get_network_field(self.kg_index, 0, 0)), self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data : dataset.TripletDataset):
        return CFKGQueryEncoder(train_data, self.embed_dim, self.rel_emb, self.fuid)

    def _get_neg_emb(self, query, pos_id):
        pos_prob, neg_id, neg_prob = self.sampler(query, self.neg_count, pos_id)
        return pos_prob, self.item_encoder(neg_id), neg_prob
    
    def _get_neg_kg_emb(self, query, pos_id):
        pos_prob, neg_id, neg_prob = self.kg_sampler(query, self.neg_count, pos_id)
        return pos_prob, self.item_encoder(neg_id), neg_prob
    
    def _get_neg_user_emb(self, query, pos_id):
        pos_prob, neg_id, neg_prob = self.user_sampler(query, self.neg_count, pos_id)
        return pos_prob, self.query_encoder(neg_id, add_inter_emb=False), neg_prob

    def forward(self, batch_data, full_score):
        user_e = self.query_encoder(batch_data[self.fuid], add_inter_emb=False) # [batch_size, dim]
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
        tail_score = {'pos_score' : tail_pos_score, 'log_pos_prob' : torch.cat([pos_item_prob, pos_t_prob]), 
                      'neg_score' : tail_neg_score, 'log_neg_prob' : torch.cat([neg_item_prob, neg_t_prob])}
        # replace head to get corrupt triple
        query = torch.cat([item_e, t_e])
        pos = torch.cat([user_e, h_e]) + torch.cat([inter_e, r_e])
        neg = torch.cat([neg_user_e, neg_h_e]) + torch.cat([inter_e, r_e]).unsqueeze(1)
        head_pos_score = self.score_func(query, pos)
        head_neg_score = self.score_func(query, neg)
        head_score = {'pos_score' : head_pos_score, 'log_pos_prob' :torch.cat([pos_user_prob, pos_h_prob]), 
                      'neg_score' : head_neg_score, 'log_neg_prob' : torch.cat([neg_user_prob, neg_h_prob])}
        return {'tail_score' : tail_score, 'head_score' : head_score}

    def training_step(self, batch, batch_idx):
        r""" Training for a batch."""
        output = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        tail_score, head_score = output['tail_score'], output['head_score']
        return self.loss_fn(None, **tail_score) + self.loss_fn(None, **head_score)

    def get_item_vector(self):
        return self.item_encoder.weight[1 : self.num_items]