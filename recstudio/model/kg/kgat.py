from os import name
from numpy.lib.function_base import select
import scipy as sp
from recstudio.model import basemodel, loss_func, scorer
from recstudio.model.kg.loops import KGATFitLoop
from recstudio.data import dataset
from recstudio.ann import sampler 
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from recstudio.model.kg import KGLearning
from recstudio.model.module import aggregator

class KGAT_TransRTower(KGLearning.TransRTower):
    def config_loss(self):
        return loss_func.BPRLoss()

class KGATConv(nn.Module):
    def __init__(self, weight_size_list, mess_dropout, alg_type):
        super().__init__()
        self.weight_size_list = weight_size_list
        self.mess_dropout = mess_dropout
        if alg_type == 'bi':
            self.aggregator_class = aggregator.BiAggregator
        elif alg_type == 'gcn':
            self.aggregator_class = aggregator.GCNAggregator
        elif alg_type == 'graphsage':
            self.aggregator_class = aggregator.GraphSageAggregator
        self.alg_type = alg_type
        self.aggregators = torch.nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.weight_size_list[ : -1], self.weight_size_list[1 : ])):
            self.aggregators.append(self.aggregator_class(input_size, output_size, dropout=self.mess_dropout[i], act=nn.LeakyReLU()))

    def forward(self, A_in, embeddings):
        all_embeddings = [embeddings]
        for aggregator in self.aggregators:
            # [num_users + num_entities, dim]
            side_embeddings = torch.sparse.mm(A_in, all_embeddings[-1])
            embeddings = aggregator(embeddings, side_embeddings)
            norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(norm_embeddings)
        return all_embeddings

class KGATItemEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = None
    
    def forward(self, batch_data):
        return self.item_embeddings[batch_data]

class KGATIUserEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embeddings = None
    
    def forward(self, batch_data):
        return self.user_embeddings[batch_data]
"""
KGAT
##################
    KGAT: Knowledge Graph Attention Network for Recommendation(KDD'19)
    Reference:
        https://dl.acm.org/doi/10.1145/3292500.3330989
"""
class KGAT(basemodel.TwoTowerRecommender):
    """
    In KGAT, based on the item-entity alignment set, the user-item graph is seamlessly integrated with KG as a unified graph, named CKG. 
    And then, on CKG, KGAT uses GNN-based architecture and an attention mechanism to explicitly models the high-order connectivities in KG  in an end-to-end fashion.
    TransR is also used in KGAT to model the entities and relations on the granularity of triples, working as a regularizer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.kg_index = config['kg_network_index']
        self.alg_type = config['alg_type']
        self.mess_dropout = config['mess_dropout']
        self.weight_size_list = [self.embed_dim] + config['layer_size']

    def init_model(self, train_data):
        self.fhid = train_data.get_network_field(self.kg_index, 0, 0)
        self.ftid = train_data.get_network_field(self.kg_index, 0, 1)
        self.frid = train_data.get_network_field(self.kg_index, 0, 2)
        self.A_size = (train_data.num_users + train_data.num_values(self.fhid), train_data.num_users + train_data.num_values(self.fhid))
        self.ckg, num_relations = train_data.get_graph(idx=[0, 2], form='dgl', value_fields=['inter', 'relation_id'], bidirectional=[True, True], \
            row_offset=[0, train_data.num_users], col_offset=[train_data.num_users, train_data.num_users], shape=(self.A_size, self.A_size))
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.num_entities = train_data.num_values(self.fhid)
        self.num_relations = num_relations
        self.user_emb = nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

        self.all_h = self.ckg.edges()[0]
        self.all_t = self.ckg.edges()[1]
        self.all_r = self.ckg.edata['value']
        # KG Tower
        self.TransRTower = KGAT_TransRTower(self.config)
        self.TransRTower.init_model(train_data)
        self.TransRTower.rel_emb = nn.Embedding(self.num_relations, self.embed_dim, padding_idx=0)
        self.TransRTower.pro_matrix_emb = nn.Embedding(self.num_relations, self.TransRTower.pro_embed_dim * self.embed_dim, padding_idx=0)

        # initialize the attentive matrix A in CF phase
        self.A_in = self.init_att_using_lap('si')
        # define GNN 
        self.KGATConv = KGATConv(self.weight_size_list, self.mess_dropout, self.alg_type)
        super().init_model(train_data)

    def get_dataset_class(self):
        return dataset.MFDataset

    def set_train_loaders(self, train_data):
        train_data.loaders = [train_data.loader, train_data.network_feat[self.kg_index].loader]
        return False

    def config_fitloop(self, trainer):
        trainer.fit_loop = KGATFitLoop()

    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def config_loss(self):
        return loss_func.BPRLoss()

    def build_item_encoder(self, train_data):
        return KGATItemEncoder()

    def build_user_encoder(self, train_data):
        return KGATIUserEncoder()

    def init_att_using_lap(self, norm):
        """
        Initialize attention matrix using Laplacian matrix.
        
        Args:
            norm(str): which way to normalize the adjacency matrix.
        Returns:
            A_in_final(torch.Tensor): the initialized adjacency matrix stored in sparse formats. 
        """
        import dgl
        def _bi_norm_lap(adj): 
            """
            Generate bi-normalized adjacency matrix.
            """
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.sparse.diags(d_inv_sqrt)
            norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def _si_norm_lap(adj):
            """
            Generate si-normalized adjacency matrix.
            """
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.sparse.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        lap_list = []
        for i in range(1, self.num_relations, 1):
            selected_edge_idx = self.ckg.filter_edges(lambda edges: edges.data['value'] == i)
            sub_graph = dgl.edge_subgraph(self.ckg, selected_edge_idx, relabel_nodes=False)
            adj = sub_graph.adj(scipy_fmt='coo').astype(float)
            if norm == 'si':
                lap = _si_norm_lap(adj) 
            elif norm == 'bi':
                lap = _bi_norm_lap(adj)
            else:
                raise ValueError(f'adjacency matrix normlize method [{norm}] is not supported')
            lap_list.append(lap)

        A_in = sum(lap_list).tocoo()
        A_in_final = torch.sparse_coo_tensor(np.stack([A_in.row, A_in.col]), A_in.data, self.A_size, dtype=torch.float)
        return A_in_final
    
    def update_encoders(self):
        self.A_in = self.A_in.to(self.device)
        embeddings = torch.cat([self.user_emb.weight, self.TransRTower.ent_emb.weight], dim=0)
        all_embeddings = self.KGATConv(self.A_in, embeddings)
        # [num_users + num_entities, dim * num_agg]
        all_embeddings = torch.cat(all_embeddings, dim=-1)
        self.user_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_entities], dim=0)

    def forward(self, batch_data, full_score):
        self.update_encoders()
        return super().forward(batch_data, full_score)

    def training_step(self, batch, batch_idx):
        if self.fhid in batch:
            batch[self.frid] += 3 # add offset(three relations : [PAD], interaction, inv_interaction)
            y_h = self.TransRTower.forward(batch)
            loss = self.TransRTower.loss_fn(None, *y_h)
            return {'loss' : loss, 'loss_kg' : loss.detach()}
        else:
            y_h = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
            loss = self.loss_fn(None, *y_h)
            return {'loss' : loss, 'loss_rec': loss.detach()}

    def _generate_transE_score(self, all_h, all_t, r):
        embeddings = torch.cat([self.user_emb.weight, self.TransRTower.ent_emb.weight], dim=0)
        # [all_h, embed_dim]
        h_e = embeddings[all_h] 
        t_e = embeddings[all_t]
        # [pro_embed_dim]
        r_e = self.TransRTower.rel_emb.weight[r]
        # [embed_dim, pro_embed_dim] to reduce memory
        pro_matrix = self.TransRTower.pro_matrix_emb.weight[r].reshape(self.embed_dim, self.TransRTower.pro_embed_dim)
        # [all_h, pro_embed_dim]
        h_e = torch.matmul(h_e, pro_matrix)
        t_e = torch.matmul(t_e, pro_matrix)

        kg_score = torch.sum(t_e * torch.tanh(h_e + r_e), dim=-1)
        return kg_score

    def update_attentive_A(self):
        self.all_h = self.all_h.to(self.device)
        self.all_r = self.all_r.to(self.device)
        self.all_t = self.all_t.to(self.device)
        kg_score = []
        rows = []
        cols = []
        for r in range(1, self.num_relations, 1):
            selected_flag = (self.all_r == r)
            h = self.all_h[selected_flag]
            t = self.all_t[selected_flag]
            r_kg_score = self._generate_transE_score(h, t, r)
            kg_score.append(r_kg_score)
            rows.append(h)
            cols.append(t)
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        kg_score = torch.cat(kg_score)
        A_in = torch.sparse_coo_tensor(torch.stack([rows, cols]), kg_score, self.A_size)
        A_in = torch.sparse.softmax(A_in, dim=1)
        self.A_in = A_in

    def on_train_epoch_start(self):
        pass

    def training_epoch_end(self, outputs):
        for output in outputs:
            output.pop('loss')
        return super().training_epoch_end(outputs)

    def prepare_testing(self):
        with torch.no_grad():
            self.update_attentive_A()
        self.update_encoders()
        self.register_buffer('item_vector', self.item_encoder.item_embeddings[1 : self.num_items].detach().clone())
        if self.use_index:
            self.ann_index = self.build_ann_index()
