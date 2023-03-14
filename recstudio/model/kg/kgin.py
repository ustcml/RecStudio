from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn 
from recstudio.ann import sampler
from recstudio.model import basemodel, scorer, loss_func
from recstudio.data.dataset import TripletDataset
from recstudio.model.module import graphmodule

class KGINAggregator(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def _get_kg_message_func(self):
        return fn.e_mul_u('rel_h', 'ent_h', 'msg')

    def _get_kg_reduce_func(self):
        return fn.mean('msg', 'neigh')

    def forward(self, knowledge_graph, interact_mat, ent_emb, user_emb, latent_emb, rel_emb, disen_att):
        # KG
        knowledge_graph.update_all(self._get_kg_message_func(), self._get_kg_reduce_func())
        # calculate user -> latent factor attentions
        score = torch.mm(user_emb, latent_emb.T).unsqueeze(dim=-1) # [num_users, num_factors, 1]
        # user aggregate 
        user_agg = torch.sparse.mm(interact_mat, ent_emb[:interact_mat.size(-1)]) # [num_users, dim]
        # calculate latent factor embeddings 
        disen_embeddings = torch.mm(F.softmax(disen_att, dim=-1), rel_emb) # [num_factors, dim]
        disen_embeddings = disen_embeddings.expand(user_agg.size(0), -1, -1) # [num_users, num_factors, dim]
        user_agg = user_agg * (disen_embeddings * score).sum(dim=-2) + user_agg 
        
        return knowledge_graph.ndata['neigh'], user_agg

class KGINGraphConv(torch.nn.Module):
    
    def __init__(self, config, num_users, num_entities, num_relations) -> None:
        super().__init__()
        self.config = config
        self.num_users = num_users
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.num_layers = self.config['model']['num_layers']
        self.embed_dim = self.config['model']['embed_dim']
        self.num_factors = self.config['model']['num_factors']     
        self.intents_indep = self.config['model']['intents_indep']
        self.temperature = 0.2

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(KGINAggregator())

        self.disen_weight_att = nn.init.xavier_normal(torch.empty(self.config['model']['num_factors'], self.num_relations))
        self.disen_weight_att = nn.parameter.Parameter(self.disen_weight_att) # [num_factors, num_relations]

        self.node_dropout = graphmodule.EdgeDropout(self.config['model']['node_dropout']) if (self.config['model']['node_dropout'] is not None) else None 
        self.mess_dropout = torch.nn.Dropout(self.config['model']['mess_dropout']) if (self.config['model']['mess_dropout'] is not None) else None 

    def _cul_cor(self):
        
        def CosineSimilarity():
            normalized_disen_weight_att = F.normalize(self.disen_weight_att, p=2, dim=-1) # [num_factors, num_relations]
            logits = torch.mm(normalized_disen_weight_att, normalized_disen_weight_att.T) 
            return logits.sum() / 2

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2 : [dim]
            dim = tensor_1.shape[0]
            zeros = torch.zeros(dim, dim).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1) # [dim, 1]
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [dim, dim]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [dim, dim]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / dim ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / dim ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / dim ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            loss_func = nn.CrossEntropyLoss(reduction='sum')
            normalized_disen_weight_att = F.normalize(self.disen_weight_att, p=2, dim=-1)
            logits = torch.mm(normalized_disen_weight_att, normalized_disen_weight_att.T) / self.temperature # [num_factors, num_factors]
            labels = torch.arange(self.disen_weight_att.size(0)) # [num_factors]
            loss = loss_func(logits, labels)
            return loss 
        
        if self.config['model']['intents_indep'] == 'mi':
            return MutualInformation()
        elif self.config['model']['intents_indep'] == 'cosine':
            return CosineSimilarity()
        else:
            cor = 0
            for i in range(self.num_factors):
                for j in range(i + 1, self.num_factors):
                    cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
            return cor

    def forward(self, knowledge_graph, interact_mat, user_emb, entity_emb, latent_emb, rel_emb):
        # node dropout
        if self.node_dropout is not None:
            knowledge_graph = self.node_dropout(knowledge_graph)
            interact_mat = self.node_dropout(interact_mat)

        entity_res_emb = entity_emb
        user_res_emb = user_emb 
        cor = self._cul_cor()
        knowledge_graph.ndata['ent_h'] = entity_emb
        knowledge_graph.edata['rel_h'] = rel_emb[knowledge_graph.edata['value']]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](knowledge_graph, interact_mat, entity_emb, user_emb, latent_emb, rel_emb, self.disen_weight_att)
            # mess dropout 
            if self.mess_dropout is not None: 
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            knowledge_graph.ndata['ent_h'] = entity_emb
            
            entity_res_emb = entity_res_emb + entity_emb
            user_res_emb = user_res_emb + user_emb

        return entity_res_emb, user_res_emb, cor 


class KGIN(basemodel.BaseRetriever):
    
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.kg_index = self.config['data']['kg_network_index']

    def _init_model(self, train_data: TripletDataset):

        # dataset property
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.frid = train_data.get_network_field(self.kg_index, 0, 1)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 2)
        super()._init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items 
        self.num_entities = train_data.num_values(self.fhid)
        self.num_factors = self.config['model']['num_factors']

        # graph
        self.interact_mat = self.get_bi_norm_adj_mat(train_data) # [num_users, num_items]
        self.knowledge_graph, self.num_relations = train_data.get_graph(idx=[self.kg_index+1], form='dgl', value_fields=['relation_id'], bidirectional=[True], \
            shape=(self.num_entities, self.num_entities))
        self.knowledge_graph = self.knowledge_graph

        # embeddings 
        self.user_emb = torch.nn.Embedding(self.num_users, self.embed_dim, padding_idx=0)
        self.ent_emb = torch.nn.Embedding(self.num_entities, self.embed_dim, padding_idx=0)
        self.latent_emb = torch.nn.Embedding(self.num_factors, self.embed_dim, padding_idx=0) # using for calculate user -> latent factor attentions. 
        self.rel_emb = torch.nn.Embedding(self.num_relations, self.embed_dim, padding_idx=0) # not include interact relation.

        # graph convolution module
        self.gcn = KGINGraphConv(self.config, self.num_users, self.num_entities, self.num_relations)

    def get_bi_norm_adj_mat(self, train_data: TripletDataset):
        """
        Get a binary normlized adjacency matrix as the author did in source code.
        Get the binary normlized adjacency matrix following the formula:
        
        .. math::
            norm_adj = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
        
        Returns:
            norm_adj(tensor): the binary normlized adjacency matrix in COO format.
        """
        interaction_matrix, _ = train_data.get_graph([0], value_fields='inter')
        adj_size = train_data.num_users + train_data.num_items
        rows = np.concatenate([interaction_matrix.row, interaction_matrix.col + train_data.num_users])
        cols = np.concatenate([interaction_matrix.col + train_data.num_users, interaction_matrix.row])
        vals = np.ones(len(rows))
        adj_mat = sp.coo_matrix((vals, (rows, cols)), shape=(adj_size, adj_size))
        rowsum = np.array(adj_mat.sum(axis=-1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv).tocsr()
        norm_adj = norm_adj[:self.num_users, self.num_users:].tocoo()
        row, col, data = torch.from_numpy(norm_adj.row), torch.from_numpy(norm_adj.col), torch.from_numpy(norm_adj.data)
        norm_adj = torch.sparse_coo_tensor(np.stack([row, col]), data, (self.num_users, self.num_items), dtype=torch.float)
        return norm_adj

    def _get_dataset_class():
        return TripletDataset
    
    def _set_data_field(self, data):
        fhid = data.get_network_field(self.kg_index, 0, 0)
        frid = data.get_network_field(self.kg_index, 0, 1)
        ftid = data.get_network_field(self.kg_index, 0, 2)
        data.use_field = set([data.fuid, data.fiid, data.frating, fhid, frid, ftid])

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items - 1)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.BPRLoss()

    def _get_query_encoder(self, train_data):
        return graphmodule.GraphUserEncoder()

    def _get_item_encoder(self, train_data):
        return graphmodule.GraphItemEncoder()

    def update_encoders(self):
        self.knowledge_graph = self.knowledge_graph.to(self._parameter_device)
        self.interact_mat = self.interact_mat.to(self._parameter_device)
        entity_gcn_emb, user_gcn_emb, cor = \
            self.gcn(self.knowledge_graph, self.interact_mat, 
                     self.user_emb.weight, self.ent_emb.weight, 
                     self.latent_emb.weight, self.rel_emb.weight)
        self.query_encoder.user_embeddings = user_gcn_emb
        self.item_encoder.item_embeddings = entity_gcn_emb
        return cor

    def forward(self, batch):
        cor = self.update_encoders()
        output = super().forward(batch, False, return_query=True, return_item=True)
        output['indents_indep_loss'] = cor
        return output

    def training_step(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(None, **output['score']) + self.config['model']['l2_reg'] * loss_func.l2_reg_loss_fn(output['query'], output['item']) \
            + self.config['model']['sim_regularity'] * output['indents_indep_loss']  
        return loss

    def _get_item_vector(self):
        if self.item_encoder.item_embeddings == None:
            return self.ent_emb.weight[1 : self.num_items].detach().clone()
        else:
            return self.item_encoder.item_embeddings[1 : self.num_items].detach().clone()

    def _update_item_vector(self):
        self.update_encoders()
        super()._update_item_vector()





