from torchrec.model.kg.layers import MLPModule
from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
from torch import nn
import torch.nn.functional as F
import torch
import collections
import numpy as np

class Aggregator(nn.Module):
    def __init__(self, dim, dropout=0, act=torch.nn.ReLU(), aggregator_type='sum'):
        super().__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim
        self.aggregator_type = aggregator_type
        assert self.aggregator_type in {'sum', 'concat', 'neighbor'}
        if self.aggregator_type == 'concat':
            self.linear = nn.Linear(self.dim * 2, self.dim, bias=True)
        else:
            self.linear = nn.Linear(self.dim, self.dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.reshape(-1, 1, 1, self.dim)
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_score = torch.mean(neighbor_relations * user_embeddings, dim=-1)
        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_ralation_score_normalized = torch.softmax(user_relation_score, dim=-1).unsqueeze(-1)
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim] 
        neighbors_aggregated = torch.mean(user_ralation_score_normalized * neighbor_vectors, dim=-2)
        return neighbors_aggregated

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        if self.aggregator_type == 'sum':
            output = self.dropout(self_vectors + neighbors_agg)
        elif self.aggregator_type == 'concat':
            output = self.dropout(torch.cat([self_vectors, neighbors_agg], dim=-1))
        elif self.aggregator_type == 'neighbor':
            output = self.dropout(neighbors_agg)
        # [batch_size, -1, dim]
        output = self.linear(output) 
        return self.act(output)

class KGCNItemEncoder(nn.Module):
    def __init__(self, ent_emb, rel_emb, adj_entity, adj_relation, config):
        super().__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.n_iter = config['n_iter']
        self.n_neighbor = config['neighbor_sample_size']
        self.dim = config['embed_dim']
        self.aggregator_type = config['aggregator_type']
        self.aggregators = torch.nn.ModuleList()
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                self.aggregators.append(Aggregator(self.dim, act=torch.nn.Tanh(), aggregator_type=self.aggregator_type))
            else:
                self.aggregators.append(Aggregator(self.dim, aggregator_type=self.aggregator_type))

    def _aggregate(self, entities, relations, user_embeddings):
        # entity_vector: [batch_size, -1, dim]
        entity_vectors = [self.ent_emb(i) for i in entities]
        relation_vectors = [self.rel_emb(i) for i in relations]
        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [entity_vectors[hop].size(0), -1, self.n_neighbor, self.dim]
                vector = self.aggregators[i](entity_vectors[hop], 
                                        entity_vectors[hop + 1].reshape(shape),
                                        relation_vectors[hop].reshape(shape),
                                        user_embeddings)
                entity_vectors_next_iter.append(vector)

            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].reshape(-1, self.dim)

    def forward(self, entities, relations, user_embeddings):
        item_embeddings = self._aggregate(entities, relations, user_embeddings)
        return item_embeddings

class KGCN(basemodel.TwoTowerRecommender):
    def __init__(self, config):
        self.n_iter = config['n_iter']
        self.neighbor_sample_size = config['neighbor_sample_size']
        self.n_neighbor = self.neighbor_sample_size
        self.aggregator_type = config['aggregator_type']
        super().__init__(config)

    def init_model(self, train_data):
        self.fhid = train_data.fhid
        self.ftid = train_data.ftid
        self.frid = train_data.frid
        self.num_entities = train_data.num_entities
        self.num_items = train_data.num_items
        self.kg = self._construct_kg(train_data.kg_feat)
        self.adj_entity, self.adj_relation = self._construct_adj()
        self.ent_emb = basemodel.Embedding(train_data.num_entities, self.embed_dim, padding_idx=0)
        self.rel_emb = nn.Embedding(train_data.num_relations, self.embed_dim, padding_idx=0)     
        super().init_model(train_data)
        
    def get_dataset_class(self):
        return dataset.KnowledgeBasedDataset

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def config_loss(self):
        return nn.BCEWithLogitsLoss()

    def build_item_encoder(self, train_data):
        return KGCNItemEncoder(self.ent_emb, self.rel_emb, self.adj_entity, self.adj_relation, self.config)

    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def construct_query(self, batch_data):
        return self.user_encoder(batch_data[self.fuid])

    def _construct_kg(self, kg_feat):
        """
        Args:
            kg_feat(TensorFrame): Knowledge graph feature.
        Returns:
            kg(defaultidict): The key is head_id, and the value is a list. The list contains all triplets whose heads are head_id.
        """
        # head -> [(tail, relation), (tail, relation), ..., (tail, relation)]
        kg = collections.defaultdict(list)
        for i in range(len(kg_feat)):
            row = kg_feat[i]
            head_id = row[self.fhid].item()
            tail_id = row[self.ftid].item()
            relation_id = row[self.frid].item()
            kg[head_id].append((tail_id, relation_id))
            kg[tail_id].append((head_id, relation_id)) # treat KG as an undirected graph
        return kg

    def _construct_adj(self):
        # each line of adj_entity stores the sampled neighbor entities for a given entity
        # each line of adj_relation stores the corresponding sampled neighbor relations
        # TODO if the entity has no neighbors? 
        adj_entity = np.zeros([self.num_entities, self.neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([self.num_entities, self.neighbor_sample_size], dtype=np.int64)
        for entity in range(self.num_entities):
            neighbors = self.kg[entity]
            if neighbors == []:
                adj_entity[entity] = np.array([entity] * self.neighbor_sample_size)
                continue
            n_neighbors = len(neighbors)
            if n_neighbors >= self.neighbor_sample_size:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=self.neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=self.neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        return torch.from_numpy(adj_entity), torch.from_numpy(adj_relation)

    def _get_neighbors(self, seeds):
        self.adj_entity = self.adj_entity.to(self.device)
        self.adj_relation = self.adj_relation.to(self.device)
        seeds = seeds.unsqueeze(dim=-1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = self.adj_entity[entities[i]].reshape(seeds.size(0), -1)
            neighbor_relations = self.adj_relation[entities[i]].reshape(seeds.size(0), -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def forward(self, batch_data):
        pos_query = self.construct_query(batch_data) # [batch_size, dim]
        pos_items = self.get_item_feat(batch_data)
        # get negative items, neg_item_idx : [batch_size, neg]
        pos_prob, neg_item_idx, neg_prob = self.sampler(pos_query, self.neg_count, pos_items)
        neg_item_idx = neg_item_idx.flatten()
        neg_user_idx = batch_data[self.fuid].repeat_interleave(self.neg_count)
        batch_data[self.fuid] = torch.cat([batch_data[self.fuid], neg_user_idx])
        batch_data[self.fiid] = torch.cat([batch_data[self.fiid], neg_item_idx])

        query = self.construct_query(batch_data)
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self._get_neighbors(batch_data[self.fiid])
        item_embeddings = self.item_encoder(entities, relations, query)
        y_h = self.score_func(query, item_embeddings)

        y = torch.zeros(query.size(0)).to(self.device)
        y[ : len(pos_query)] = 1.

        return y, y_h

    def training_step(self, batch, batch_idx):
        y, y_h = self.forward(batch)
        loss = self.loss_fn(y_h, y)
        return loss

    def get_item_vector(self):
        return None

    def on_train_epoch_start(self):
        pass

    def prepare_testing(self):
        pass

    def _test_step(self, batch, metric, cutoffs):
        items = torch.arange(1, self.num_items).to(self.device)
        items = items.tile(batch[self.fuid].size(0))
        users = batch[self.fuid].repeat_interleave(self.num_items - 1)
        user_embeddings = self.user_encoder(users)
        entities, relations = self._get_neighbors(items)
        self.item_vector = self.item_encoder(entities, relations, user_embeddings).reshape(-1, self.num_items - 1, self.embed_dim)
        return super()._test_step(batch, metric, cutoffs)

# TODO It is hard to get item_vector on train_epoch_start and prepare_testing, because item_vector depends on user_embeddings