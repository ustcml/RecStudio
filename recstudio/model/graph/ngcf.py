from recstudio.data.dataset import TripletDataset
from recstudio.model import basemodel, loss_func, scorer
from recstudio.ann import sampler
import torch
from recstudio.model.module import graphmodule

r"""
NGCF
#############
    Neural Graph Collaborative Filtering (SIGIR'19)
    Reference:
        https://dl.acm.org/doi/10.1145/3331184.3331267
"""
class NGCF(basemodel.BaseRetriever):
    r"""
    NGCF is a GNN-based model, which exploits the user-item graph structure by propagating embeddings on it.
    It can model high-order connectivity expressively in user-item graph in an explict manner.
    We implement NGCF by BiAggregator.
    """
    def __init__(self, config):
        super().__init__(config)
        self.layers = config['model']['layer_size']
        self.mess_dropout = config['model']['mess_dropout']
        self.node_dropout = config['model']['node_dropout']

    def _init_model(self, train_data: TripletDataset):
        super()._init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.user_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

        self.combiners = torch.nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.layers[ : -1], self.layers[1 : ])):
            self.combiners.append(graphmodule.BiCombiner(input_size, output_size, dropout=self.mess_dropout[i], act=torch.nn.LeakyReLU()))
        self.NGCFNet = graphmodule.LightGCNNet_dglnn(self.combiners, normalize=2, mess_norm='left')

        adj_size = train_data.num_users + train_data.num_items
        self.adj_mat, _ = train_data.get_graph([0], form='dgl', value_fields='inter', \
            col_offset=[train_data.num_users], bidirectional=[True], shape=[adj_size, adj_size])

        if self.node_dropout != None:
            self.sparseDropout = graphmodule.EdgeDropout(self.node_dropout)

    def _get_dataset_class():
        return TripletDataset

    def _get_loss_func(self):
        return loss_func.BPRLoss()

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)

    def _get_query_encoder(self, train_data):
        return graphmodule.GraphUserEncoder()

    def _get_item_encoder(self, train_data):
        return graphmodule.GraphItemEncoder()

    def update_encoders(self):
        self.adj_mat = self.adj_mat.to(self.device)
        if self.node_dropout != None:
            # node dropout
            adj_mat = self.sparseDropout(self.adj_mat)
        else:
            adj_mat = self.adj_mat
        # [num_users + num_items, dim]
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embeddings = self.NGCFNet(adj_mat, embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=-1)
        self.query_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

    def forward(self, batch_data, full_score, return_neg_id=True):
        self.update_encoders()
        return super().forward(batch_data, full_score, return_neg_id=return_neg_id)

    def training_step(self, batch):
        output = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss), True)
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
            + self.config['model']['l2_reg_weight'] * loss_func.l2_reg_loss_fn(self.user_emb(batch[self.fuid]), self.item_emb(batch[self.fiid]), \
            self.item_emb(output['neg_id'].reshape(-1)))
        return loss_value

    def _get_item_vector(self):
        if self.item_encoder.item_embeddings == None:
            return self.item_emb.weight[1:].detach().clone()
        else:
            return self.item_encoder.item_embeddings[1:].detach().clone()

    def _update_item_vector(self):
        self.update_encoders()
        super()._update_item_vector()
