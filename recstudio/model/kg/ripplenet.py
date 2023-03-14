from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler 
from torch import nn
import torch.nn.functional as F
import torch
import collections
import numpy as np

"""
RippleNet
    RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems(CIKM'18)
    Reference:
        https://doi.org/10.1145/3269206.3271739
"""
class RippleNet(basemodel.BaseRanker):
    """
    RippleNet introduces the preference propagation mechanism in the KG. The h-hop neighbors and relations in h-th layer triplet set and candidate item are used to calculate the weight
    for tail entities in the corresponding triplet. And then user's representation in the h-th layer is be calculated with the weighted average of tail entity embeddings. 
    """
    def __init__(self, config):
        self.kg_index = config['data']['kg_network_index']
        self.n_hop = config['model']['n_hop']
        self.n_memory = config['model']['n_memory']
        self.item_update_mode  = config['model']['item_update_mode']
        self.using_all_hops = config['model']['using_all_hops']
        self.kge_weight = config['model']['kge_weight']
        super().__init__(config)
    
    def _init_model(self, train_data: dataset.TripletDataset):
        super()._init_model(train_data)
        
        # rec field
        self.fuid = train_data.fuid
        self.fiid = train_data.fiid
        self.num_items = train_data.num_items
        
        # kg field
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 1)
        self.frid = train_data.get_network_field(self.kg_index, 0, 2)
        
        # embeddings 
        self.ent_emb = nn.Embedding(train_data.num_values(train_data.get_network_field(self.kg_index, 0, 0)), self.embed_dim, padding_idx=0)
        self.rel_emb = nn.Embedding(train_data.num_values(self.frid), self.embed_dim * self.embed_dim, padding_idx=0)
        self.score_func = scorer.InnerProductScorer()

        # get ripple_set_h, ripple_set_t and ripple_set_r
        # remember to get the sub inter_feat that only includes train data 
        self.user_history_dict = self._construct_user_hist(train_data.inter_feat[train_data.inter_feat_subset]) 
        self.kg = self._construct_kg(train_data.network_feat[self.kg_index])
        self.ripple_set_h = torch.zeros(self.n_hop, train_data.num_users, self.n_memory, dtype=torch.long)
        self.ripple_set_t = torch.zeros(self.n_hop, train_data.num_users, self.n_memory, dtype=torch.long)
        self.ripple_set_r = torch.zeros(self.n_hop, train_data.num_users, self.n_memory, dtype=torch.long)
        self._get_ripple_set()
        if self.item_update_mode == 'replace_transform' or self.item_update_mode == 'plus_transform':
            self.transform_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
    def _get_dataset_class():
        return dataset.TripletDataset
    
    def _set_data_field(self, data):
        fhid = data.get_network_field(self.kg_index, 0, 0)
        frid = data.get_network_field(self.kg_index, 0, 1)
        ftid = data.get_network_field(self.kg_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, fhid, frid, ftid])

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
        
    def _construct_user_hist(self, inter_feat):
        user_history_dict = collections.defaultdict(list)
        for i in range(len(inter_feat[self.fuid])):
            if inter_feat[self.frating][i] > 0: 
                user = inter_feat[self.fuid][i].item()
                item = inter_feat[self.fiid][i].item()
                user_history_dict[user].append(item)
        return user_history_dict

    def _construct_kg(self, kg_feat):
        """
        Construct knowledge graph dict. 

        Args:
            kg_feat(TensorFrame): the triplets in knowledge graph.

        Returns:
            kg(defaultidict): the key is ``head_id``, and the value is a list. The list contains all triplets whose heads are ``head_id``.
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
        Gets ripple set of users in every hop.

        Returns:
            ripple_set_h(torch.tensor): shape: (n_hop, num_users)
            ripple_set_h[h][user] stores the head entities in user's h-th hop ripple set.
            ripple_set_r(torch.tensor): shape: (n_hop, num_users)
            ripple_set_r[h][user] stores the relations in user's h-th hop ripple set.
            ripple_set_t(torch.tensor): shape: (n_hop, num_users)
            ripple_set_t[h][user] stores the tail entities in user's h-th hop ripple set.
            (ripple_set_h[h][user][i], ripple_set_r[h][user][i], ripple_set_t[h][user][i]) forms a triplet.   
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
                    if h == 0:
                        pass 
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
    
    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memories, dim, 1]
            h = h_emb_list[hop].unsqueeze(-1)
            # [batch_size, n_memories, dim, dim]
            r = r_emb_list[hop].reshape(-1, self.n_memory, self.embed_dim, self.embed_dim)
            # [batch_size, n_memories, dim] 
            Rh = torch.matmul(r, h).squeeze(-1)
            # [batch_size, dim, 1] 
            v = self.item_embeddings.unsqueeze(-1)
            # [batch_size, n_memories]
            probs = torch.matmul(Rh, v).squeeze(-1)
            # [batch_size, n_memories]
            probs_normalized = torch.softmax(probs, dim=-1)
            # [batch_size, n_memories, 1] 
            probs_expanded = probs_normalized.unsqueeze(-1)
            # [batch_size, n_memories, dim] -> [batch_size, dim]
            o = torch.sum(t_emb_list[hop] * probs_expanded, dim=-2)
            
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
        """
        Constructs user embeddings. 
        Evaluating and training process share this function. 
        In evaluating, the scores between a user and all items should be calculated. 
        To do this, shape of memories_h should be reshaped into (batch_size, 1, n_memories) and 
        in _key_addressing method, the secend dimension will be broadcasted to num_items.
        """
        self.ripple_set_h = self._to_device(self.ripple_set_h, self._parameter_device)
        self.ripple_set_r = self._to_device(self.ripple_set_r, self._parameter_device)
        self.ripple_set_t = self._to_device(self.ripple_set_t, self._parameter_device)
        
        users = batch_data[self.fuid]
        # [n_hop, batch_size, n_memories] || [n_hop, batch_size, num_items, n_memories]
        self.memories_h = [] 
        self.memories_r = []
        self.memories_t = []
        for i in range(self.n_hop):
            # [batch_size, n_memories]
            self.memories_h.append(self.ripple_set_h[i][users])
            self.memories_r.append(self.ripple_set_r[i][users])
            self.memories_t.append(self.ripple_set_t[i][users])
        # [batch_size, dim]
        self.item_embeddings = self.ent_emb(batch_data[self.fiid]) 

        h_emb_list = [] # [n_hop, batch_size, n_memories, dim]
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch_size, n_memories, dim] || [batch_size, 1, n_memories, dim]
            h_emb_list.append(self.ent_emb(self.memories_h[i]))
            r_emb_list.append(self.rel_emb(self.memories_r[i]))
            t_emb_list.append(self.ent_emb(self.memories_t[i]))
 
        o_list = self._key_addressing(h_emb_list, r_emb_list, t_emb_list)
        query = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                query += o_list[i]

        return query, h_emb_list, r_emb_list, t_emb_list

    def score(self, batch):
        query, h_emb_list, r_emb_list, t_emb_list = self.construct_query(batch)
        y_h = self.score_func(query, self.item_embeddings)
        if self.training: 
            kge_loss = 0
            for hop in range(len(h_emb_list)):
                # [batch_size, n_memories, 1, dim]
                h = h_emb_list[hop].unsqueeze(-2)
                # [batch_size, n_memories, dim, dim]
                r = r_emb_list[hop].reshape(-1, self.n_memory, self.embed_dim, self.embed_dim)
                # [batch_size, n_memories, dim]
                hR = torch.matmul(h, r).squeeze(-2)
                # [batch_size, n_memories, dim]
                t = t_emb_list[hop]
                # [batch_size, n_memories]
                hRt = torch.sigmoid(torch.sum(hR * t, dim=-1))
                kge_loss += torch.mean(hRt)
            kge_loss = -self.kge_weight * kge_loss
        else:
            kge_loss = 0.0
        return {'score' : y_h, 'kge_loss' : kge_loss}

    def training_step(self, batch):
        y_h, output = self.forward(batch)
        loss = self.loss_fn(**y_h) + output['kge_loss']
        return loss