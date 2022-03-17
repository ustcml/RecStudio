from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
from torch import nn
import torch.nn.functional as F
import torch
import collections
import numpy as np

class RippleNet(basemodel.ItemTowerRecommender):
    
    def __init__(self, config):
        self.n_hop = config['n_hop']
        self.n_memory = config['n_memory']
        self.item_update_mode  = config['item_update_mode']
        self.using_all_hops = config['using_all_hops']
        self.kge_weight = config['kge_weight']
        super().__init__(config)
    
    def init_model(self, train_data):
        super().init_model(train_data)
        self.fuid = train_data.fuid
        self.num_items = train_data.num_items
        # kg 
        self.fhid = train_data.fhid
        self.ftid = train_data.ftid
        self.frid = train_data.frid
        self.rel_emb = nn.Embedding(train_data.num_relations, self.embed_dim * self.embed_dim, padding_idx=0)
        # get ripple_set_h, ripple_set_t and ripple_set_r
        # remember to get the sub inter_feat that only includes train data
        # maybe need to provide sub inter_feat directly in order to prevent users from using inter_feat incorrectly.  
        self.user_history_dict = self._construct_user_hist(train_data.inter_feat[train_data.inter_feat_subset]) 
        self.kg = self._construct_kg(train_data.kg_feat)
        self.ripple_set_h = torch.zeros(self.n_hop, train_data.num_users, self.n_memory, dtype=torch.long)
        self.ripple_set_t = torch.zeros(self.n_hop, train_data.num_users, self.n_memory, dtype=torch.long)
        self.ripple_set_r = torch.zeros(self.n_hop, train_data.num_users, self.n_memory, dtype=torch.long)
        self._get_ripple_set()
        
        if self.item_update_mode == 'replace_transform' or self.item_update_mode == 'plus_transform':
            self.transform_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def get_dataset_class(self):
        return dataset.KnowledgeBasedDataset

    def build_item_encoder(self, train_data):
        return basemodel.Embedding(train_data.num_entities, self.embed_dim, padding_idx=0)

    def config_loss(self):
        return nn.BCEWithLogitsLoss()

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def _construct_user_hist(self, inter_feat):
        user_history_dict = collections.defaultdict(list)
        for i in range(len(inter_feat[self.fuid])):
            user = inter_feat[self.fuid][i].item()
            item = inter_feat[self.fiid][i].item()
            user_history_dict[user].append(item)
        return user_history_dict

    def _construct_kg(self, kg_feat):
        """
        Args:
            kg_feat(TensorFrame): Knowledge graph feature.
        Returns:
            kg(defaultidict): The key is head_id, and the value is a list. 
            The list contains all triplets whose heads are head_id.
        """
        # head -> [(tail, relation), (tail, relation), ..., (tail, relation)]
        kg = collections.defaultdict(list)
        for i in range(len(kg_feat)):
            row = kg_feat[i]
            head_id = row[self.fhid].item()
            tail_id = row[self.ftid].item()
            relation_id = row[self.frid].item()
            kg[head_id].append((tail_id, relation_id))
        return kg

    def _get_ripple_set(self):
        """
        Returns:
            ripple_set(defaultdict):
        """
        # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
        for user in self.user_history_dict:
            for h in range(self.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = self.user_history_dict[user]
                else:
                    # tails_of_last_hop = ripple_set[user][-1][2]
                    tails_of_last_hop = self.ripple_set_t[h - 1][user]
                
                for entity in tails_of_last_hop:
                    if type(entity) == torch.Tensor:
                        entity = entity.item()
                    for tail_and_relation in self.kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])
                
                # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
                if len(memories_h) == 0:
                    # ripple_set[user].append(ripple_set[user][-1])
                    self.ripple_set_h[h][user] = self.ripple_set_h[h - 1][user]
                    self.ripple_set_r[h][user] = self.ripple_set_r[h - 1][user]
                    self.ripple_set_t[h][user] = self.ripple_set_t[h - 1][user]
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < self.n_memory
                    indices = np.random.choice(len(memories_h), size=self.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    memories_h = torch.LongTensor(memories_h)
                    memories_r = torch.LongTensor(memories_r)
                    memories_t = torch.LongTensor(memories_t)
                    self.ripple_set_h[h][user] = memories_h
                    self.ripple_set_r[h][user] = memories_r
                    self.ripple_set_t[h][user] = memories_t
                    # ripple_set[user].append((memories_h, memories_r, memories_t))
        
    
    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memories, dim, 1] || [batch_size, 1, n_memories, dim, 1]
            h = self.h_emb_list[hop].unsqueeze(-1)
            if self.training:
                # [batch_size, n_memories, dim, dim]
                r = self.r_emb_list[hop].reshape(-1, self.n_memory, self.embed_dim, self.embed_dim)
            else:
                # [batch_size, 1, n_memories, dim, dim]
                r = self.r_emb_list[hop].reshape(-1, 1, self.n_memory, self.embed_dim, self.embed_dim)
            # [batch_size, n_memories, dim] || [batch_size, 1, n_memories, dim]
            Rh = torch.matmul(r, h).squeeze(-1)
            # [batch_size, dim, 1] || [num_items, dim, 1]
            v = self.item_embeddings.unsqueeze(-1)
            # [batch_size, n_memories] || [batch_size, num_items, n_memories](using broadcast)
            probs = torch.matmul(Rh, v).squeeze(-1)
            # [batch_size, n_memories]
            probs_normalized = torch.softmax(probs, dim=-1)
            # [batch_size, n_memories, 1] 
            probs_expanded = probs_normalized.unsqueeze(-1)
            # [batch_size, n_memories, dim] -> [batch_size, dim]
            o = torch.sum(self.t_emb_list[hop] * probs_expanded, dim=-2)
            
            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)

        return o_list

    def update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(o + item_embeddings)
        return item_embeddings

    def construct_query(self, batch_data):
        if self.ripple_set_h.device != self.device:
            self.ripple_set_h = self.ripple_set_h.to(self.device)
            self.ripple_set_r = self.ripple_set_r.to(self.device)
            self.ripple_set_t = self.ripple_set_t.to(self.device) 

        users = batch_data[self.fuid]
        # [n_hop, batch_size, n_memories] || [n_hop, batch_size, num_items, n_memories]
        self.memories_h = [] 
        self.memories_r = []
        self.memories_t = []
        if not self.training:
            for i in range(self.n_hop):
                # [batch_size, 1, n_memories]
                self.memories_h.append(self.ripple_set_h[i][users].unsqueeze(-2))
                self.memories_r.append(self.ripple_set_r[i][users].unsqueeze(-2))
                self.memories_t.append(self.ripple_set_t[i][users].unsqueeze(-2))  
            #[num_items, dim]
            self.item_embeddings = self.item_encoder.weight[1 : self.num_items]
        else:
            for i in range(self.n_hop):
                # [batch_size, n_memories]
                self.memories_h.append(self.ripple_set_h[i][users])
                self.memories_r.append(self.ripple_set_r[i][users])
                self.memories_t.append(self.ripple_set_t[i][users])
            # [batch_size, dim]
            self.item_embeddings = self.item_encoder(batch_data[self.fiid]) 

        self.h_emb_list = [] # [n_hop, batch_size, n_memories, dim]
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch_size, n_memories, dim] || [batch_size, 1, n_memories, dim]
            self.h_emb_list.append(self.item_encoder(self.memories_h[i]))
            self.r_emb_list.append(self.rel_emb(self.memories_r[i]))
            self.t_emb_list.append(self.item_encoder(self.memories_t[i]))
        if self.training:
            self.kge_h_emb_list.extend(self.h_emb_list) # for calculating kge_loss
            self.kge_t_emb_list.extend(self.t_emb_list)
            self.kge_r_emb_list.extend(self.r_emb_list)
 
        o_list = self._key_addressing()
        query = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                query += o_list[i]

        if not self.training:
            self.item_vector = self.item_embeddings

        return query

    
    def forward(self, batch_data):
        pos_items = self.get_item_feat(batch_data)
        pos_query = self.construct_query(batch_data) #[batch_size, dim]
        pos_items_embeddings = self.item_embeddings

        # get negative items
        pos_prob, neg_item_idx, neg_prob = self.sampler(pos_query, self.neg_count, pos_items)
        # [batch_size, neg] -> [batch_size * neg]
        neg_item_idx = neg_item_idx.reshape(-1)
        neg_user_idx = batch_data[self.fuid].repeat_interleave(self.neg_count)
        neg_batch_data = {}
        neg_batch_data[self.fuid] = neg_user_idx
        neg_batch_data[self.fiid] = neg_item_idx
        neg_query = self.construct_query(neg_batch_data) # [batch_size * neg, dim]
        neg_item_embeddings = self.item_embeddings

        query = torch.cat([pos_query, neg_query])
        item_embeddings = torch.cat([pos_items_embeddings, neg_item_embeddings])
        y_h = self.score_func(query, item_embeddings)
        y = torch.zeros(query.size(0)).to(self.device)
        y[ : len(batch_data[self.fiid])] = 1
        
        return y, y_h


    def training_step(self, batch, batch_idx):
        # used to calculate kge loss
        self.kge_h_emb_list = []
        self.kge_t_emb_list = []
        self.kge_r_emb_list = []

        y, y_h = self.forward(batch)
        loss = self.loss_fn(y_h, y)

        # calculate kge loss
        kge_loss = 0
        for hop in range(len(self.kge_h_emb_list)):
            # [batch_size, n_memories, dim, 1]
            h = self.kge_h_emb_list[hop].unsqueeze(-1)
            # [batch_size, n_memories, dim, dim]
            r = self.kge_r_emb_list[hop].reshape(-1, self.n_memory, self.embed_dim, self.embed_dim)
            # [batch_size, n_memories, dim]
            Rh = torch.matmul(r, h).squeeze(-1)
            # [batch_size, n_memories, dim]
            t = self.kge_t_emb_list[hop]
            # [batch_size, n_memories]
            hRt = torch.sigmoid(torch.sum(Rh * t, dim=-1))
            kge_loss += torch.mean(hRt)
        kge_loss = -self.kge_weight * kge_loss
        
        return loss + kge_loss 
