from recstudio.model.module.layers import MLPModule
from recstudio.model import basemodel, loss_func, scorer
from recstudio.data import dataset
from recstudio.ann import sampler 
from torch import nn
import torch.nn.functional as F
import torch
import collections
import numpy as np
from recstudio.model.module import graphmodule

class KGCNConv(nn.Module):
    def __init__(self, n_iter, dim, n_neighbor, dropout, alg_type):
        super().__init__()
        self.n_iter = n_iter
        self.dim = dim
        self.n_neighbor = n_neighbor
        self.dropout = dropout
        if alg_type == 'sum':
            self.aggregator_class = graphmodule.GCNCombiner
        elif alg_type == 'concat':
            self.aggregator_class = graphmodule.GraphSageCombiner
        elif alg_type == 'neighbor':
            self.aggregator_class = graphmodule.NeighborCombiner
        self.alg_type = alg_type
        self.aggregators = torch.nn.ModuleList()
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                self.aggregators.append(self.aggregator_class(self.dim, self.dim, dropout=self.dropout, act=torch.nn.Tanh()))
            else:
                self.aggregators.append(self.aggregator_class(self.dim, self.dim, dropout=self.dropout))
        
    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        """
        Args: 
            neighbor_vectors(torch.Tensor): shape: (batch_size, n_neighbor^(i-1), n_neighbor, dim)
            embeddings of the neighbors in i-th layers. 
            neighbor_relations(torch.Tensor): shape: (batch_size, n_neighbor^(i-1), n_neighbor, dim). embeddings of the relations in i-th layers. 
            user_embeddings(torch.Tensor): shape: (batch_size, embed_dim). The embeddings of the query users used to calculate the weights of relations.
        Returns:
            neighbors_aggregated(torch.Tensor): shape: (batch_size, n_neighbor^(i-1), dim). The aggregation of neighbors and each neighbor has a different weight.
        """
        # [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.reshape(-1, 1, 1, self.dim)
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_score = torch.mean(neighbor_relations * user_embeddings, dim=-1)
        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_ralation_score_normalized = torch.softmax(user_relation_score, dim=-1).unsqueeze(-1)
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim] 
        neighbors_aggregated = torch.mean(user_ralation_score_normalized * neighbor_vectors, dim=-2)
        return neighbors_aggregated
    
    def forward(self, entity_vectors, relation_vectors, user_embeddings):
        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [entity_vectors[hop].size(0), -1, self.n_neighbor, self.dim]
                neighbors_agg = self._mix_neighbor_vectors(
                    entity_vectors[hop + 1].reshape(shape), 
                    relation_vectors[hop].reshape(shape), 
                    user_embeddings)
                vector = self.aggregators[i](entity_vectors[hop], neighbors_agg)
                entity_vectors_next_iter.append(vector)

            entity_vectors = entity_vectors_next_iter
        return entity_vectors[0].reshape(-1, self.dim)

class KGCNItemEncoder(nn.Module):
    def __init__(self, ent_emb, rel_emb, config):
        super().__init__()
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.KGCNConv = KGCNConv(config['n_iter'], config['embed_dim'], config['neighbor_sample_size'], \
             0.0, config['aggregator_type'])

    def forward(self, entities, relations, user_embeddings):
        """
        Args:
            entities(list): shape: ({[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]})
            the multi-hop neighbors of the query items.
            relations(list): shape: ({[batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]})
            the multi-hop relations of the query items.
            user_embeddings(torch.Tensor): shape: (batch_size, embed_dim)
            the embeddings of the query users used to calculate the weights of relations.
        Returns:
            item_embeddings(torch.Tensor): shape: (batch_size, embed_dim)
            the item embeddings for each user.  
        """
        # entity_vector: [batch_size, -1, dim]
        entity_vectors = [self.ent_emb(i) for i in entities]
        relation_vectors = [self.rel_emb(i) for i in relations]
        item_embeddings = self.KGCNConv(entity_vectors, relation_vectors, user_embeddings)
        return item_embeddings
"""
KGCN
#################
    Knowledge Graph Convolutional Networks for Recommender Systems(WWW'19)
    Reference:
        https://dl.acm.org/doi/10.1145/3308558.3313417

"""
class KGCN(basemodel.BaseRanker):
    """
    KGCN adopt GCN to obtain item embeddings via their neighbors in KG. 
    KGCN computes user-specific item embeddings by first applying a trainable function that identifies important knowledge graph relationships for a given user. 
    This way we transform the knowledge graph into a user-specific weighted graph and then apply a graph neural network to compute personalized item embeddings.
    """
    def __init__(self, config):
        self.kg_index = config['data']['kg_network_index']
        self.n_iter = config['model']['n_iter']
        self.neighbor_sample_size = config['model']['neighbor_sample_size']
        self.n_neighbor = self.neighbor_sample_size
        self.aggregator_type = config['model']['aggregator_type']
        super().__init__(config)

    def _init_model(self, train_data):
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 1)
        self.frid = train_data.get_network_field(self.kg_index, 0, 2)
        self.num_entities = train_data.num_values(self.fhid)
        self.num_items = train_data.num_items
        self.kg = self._construct_kg(train_data.network_feat[self.kg_index])
        self.adj_entity, self.adj_relation = self._construct_adj()
        self.user_emb = nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.ent_emb = nn.Embedding(self.num_entities, self.embed_dim, padding_idx=0)
        self.rel_emb = nn.Embedding(train_data.num_values(self.frid), self.embed_dim, padding_idx=0)     
        
        self.item_encoder = KGCNItemEncoder(self.ent_emb, self.rel_emb, self.config['model'])
        self.score_func = scorer.InnerProductScorer()
        
        super()._init_model(train_data)
        
    def _get_dataset_class():
        return dataset.TripletDataset
    
    def _set_data_field(self, data):
        fhid = data.get_network_field(self.kg_index, 0, 0)
        frid = data.get_network_field(self.kg_index, 0, 1)
        ftid = data.get_network_field(self.kg_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, fhid, frid, ftid])

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
    
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
            kg[tail_id].append((head_id, relation_id)) # treat KG as an undirected graph
        return kg

    def _construct_adj(self):
        # each line of adj_entity stores the sampled neighbor entities for a given entity
        # each line of adj_relation stores the corresponding sampled neighbor relations
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
        """
        Given items, gets their multi-hop neighbors in knowledge graph.
        
        Args:
            seeds(torch.Tensor): shape: (batch_size)
            items that needs to get neighbors
        Returns:
            entities(list): shape: ({[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]})
            the multi-hop neighbors of items. The i-th element in entities correspond to the (i-1)-th hop neighbors of the items and the first element is the seeds.
            relations(list): shape: ({[batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]})
            the multi-hop relations of items. The i-th element in relations correspond to the i-th hop relations of the items
        """
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

    def score(self, batch_data):
        query = self.user_emb(batch_data[self.fuid]) # [batch_size, dim]
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self._get_neighbors(batch_data[self.fiid])
        item_embeddings = self.item_encoder(entities, relations, query)
        score = self.score_func(query, item_embeddings)
        return {'score' : score} 