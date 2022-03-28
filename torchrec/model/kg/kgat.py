from os import name
from networkx.algorithms.cluster import triangles
from numpy.lib.function_base import select
import scipy as sp
from torch.nn.modules.container import ModuleList
from torchrec.model import basemodel, loss_func, scorer
from torchrec.model.kg.loops import KGATFitLoop
from torchrec.data import dataset
from torchrec.ann import sampler 
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Aggregator(nn.Module):
    def __init__(self, input_size, output_size, mess_out, alg_type):
        super().__init__()
        self.alg_tpye = alg_type
        if self.alg_tpye == 'bi':
            self.linear_sum = nn.Linear(input_size, output_size)
            self.linear_product = nn.Linear(input_size, output_size)
        elif self.alg_tpye == 'gcn':
            self.linear = nn.Linear(input_size, output_size)
        elif self.alg_tpye == 'graphsage':
            self.linear = nn.Linear(input_size * 2, output_size)
        else:
            raise ValueError(f'[{self.alg_tpye}] aggregator is not supported')
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(mess_out)
        

    def forward(self, A_in, embeddings):
        # A_in : [num_users + num_entities, num_users + num_entities]
        # embeddings: [num_users + num_entities, dim]
        # side_embeddings: [num_users + num_entities, dim]
        side_embeddings = torch.sparse.mm(A_in, embeddings)
        if self.alg_tpye == 'bi':
            sum_embeddings = self.act(self.linear_sum(embeddings + side_embeddings))
            bi_embeddings = self.act(self.linear_product(embeddings * side_embeddings))
            embeddings = sum_embeddings + bi_embeddings
        elif self.alg_tpye == 'gcn':
            embeddings = self.act(self.linear(embeddings + side_embeddings))
        elif self.alg_tpye == 'graphsage':
            embeddings = self.act(self.linear(torch.cat([embeddings + side_embeddings], dim=-1)))
        embeddings = self.dropout(embeddings)
        return embeddings

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

class KGAT(basemodel.TwoTowerRecommender):
    def __init__(self, config):
        super().__init__(config)
        self.alg_type = config['alg_type']
        self.kge_dim = config['kge_dim']
        self.mess_dropout = config['mess_dropout']
        self.weight_size_list = [self.embed_dim] + config['layer_size']

    def init_model(self, train_data):
        self.fhid = train_data.fhid
        self.ftid = train_data.ftid
        self.frid = train_data.frid
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.num_entities = train_data.num_entities
        self.num_relations = train_data.num_relations
        self.user_emb = nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.ent_emb = basemodel.Embedding(train_data.num_entities, self.embed_dim, padding_idx=0)
        
        self.rel_emb = nn.Embedding(train_data.num_relations, self.kge_dim, padding_idx=0)
        self.trans_W = nn.Embedding(train_data.num_relations, self.embed_dim * self.kge_dim, padding_idx=0)

        self.ckg = train_data.get_ckg_graph('dgl', True)
        self.all_h = self.ckg.edges()[0]
        self.all_t = self.ckg.edges()[1]
        self.all_r = self.ckg.edata[train_data.frid]
        # initialize the attentive matrix A in CF phase
        self.A_size = (train_data.num_users + train_data.num_entities, train_data.num_users + train_data.num_entities)
        self.A_in = self.init_att_using_lap('si')
        # define aggregators 
        self.aggregators = ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.weight_size_list[ : -1], self.weight_size_list[1 : ])):
            self.aggregators.append(Aggregator(input_size, output_size, self.mess_dropout[i], self.alg_type))
        self.kg_sampler = sampler.UniformSampler(self.num_entities - 1)
        self.kg_score_func = scorer.EuclideanScorer()
        self.kg_loss_fn = loss_func.BPRLoss()
        super().init_model(train_data)

    def get_dataset_class(self):
        return dataset.KnowledgeBasedDataset

    def set_train_loaders(self, train_data):
        train_data.loaders = [train_data.loader, train_data.network_feat[1].loader]
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
        import dgl
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.sparse.diags(d_inv_sqrt)
            norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.sparse.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        lap_list = []
        for i in range(1, self.num_relations, 1):
            selected_edge_idx = self.ckg.filter_edges(lambda edges: edges.data[self.frid] == i)
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
        embeddings = torch.cat([self.user_emb.weight, self.ent_emb.weight], dim=0)
        all_embeddings = [embeddings]
        for aggregator in self.aggregators:
            # [num_users + num_entities, dim]
            embeddings = aggregator(self.A_in, embeddings)
            norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(norm_embeddings)
        # [num_users + num_entities, dim * num_agg]
        all_embeddings = torch.cat(all_embeddings, dim=-1)
        self.user_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_entities], dim=0)

    def forward(self, batch_data, full_score):
        self.update_encoders()
        return super().forward(batch_data, full_score)

    def kg_forward(self, batch_data):
        h = batch_data[self.fhid]
        r = batch_data[self.frid]
        pos_t = batch_data[self.ftid]
        pos_prob, neg_t, neg_prob = self.kg_sampler(self.ent_emb(h), self.neg_count, pos_t)
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_inference(h, r, pos_t, neg_t)
        pos_score = self.kg_score_func(h_e + r_e, pos_t_e)
        neg_score = self.kg_score_func(h_e + r_e, neg_t_e)
        return pos_score, pos_prob, neg_score, neg_prob

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        # [batch_size, 1, emb_dim]
        h_e = self.ent_emb(h).unsqueeze(-2)
        pos_t_e = self.ent_emb(pos_t).unsqueeze(-2)
        # [batch_size, neg, emb_dim]
        neg_t_e = self.ent_emb(neg_t)
        # [batch_size, kge_dim]
        r_e = self.rel_emb(r)
        # [batch_size, emb_dim, kge_dim]
        trans_M = self.trans_W(r).reshape(-1, self.embed_dim, self.kge_dim)
        # [batch_size, 1, kge_dim] -> [batch_size, kge_dim] || [batch_size, neg, kge_dim]
        h_e = torch.matmul(h_e, trans_M).squeeze(-2)
        pos_t_e = torch.matmul(pos_t_e, trans_M).squeeze(-2)
        neg_t_e = torch.matmul(neg_t_e, trans_M)

        return h_e, r_e, pos_t_e, neg_t_e

    def training_step(self, batch, batch_idx):
        if self.fhid in batch:
            y_h = self.kg_forward(batch)
            loss = self.kg_loss_fn(None, *y_h)
            return {'loss' : loss, 'loss_kg' : loss.detach()}
        else:
            y_h = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
            loss = self.loss_fn(None, *y_h)
            return {'loss' : loss, 'loss_rec': loss.detach()}

    def _generate_transE_score(self, all_h, all_t, r):
        embeddings = torch.cat([self.user_emb.weight, self.ent_emb.weight], dim=0)
        # [all_h, embed_dim]
        h_e = embeddings[all_h] 
        t_e = embeddings[all_t]
        # [kge_dim]
        r_e = self.ent_emb.weight[r]
        # [embed_dim, kge_dim] to reduce memory
        trans_M = self.trans_W.weight[r].reshape(self.embed_dim, self.kge_dim)
        # [all_h, kge_dim]
        h_e = torch.matmul(h_e, trans_M)
        t_e = torch.matmul(t_e, trans_M)

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


# TODO figure out whether the ckg is a bidirectional graph
# TODO KG data doesn't includes user-item interaction.