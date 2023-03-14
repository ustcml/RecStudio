import torch
import torch.nn.functional as F
from recstudio.data import TripletDataset
from recstudio.model import basemodel, loss_func, scorer
from recstudio.model.module import data_augmentation, graphmodule
from recstudio.ann import sampler


class RandomWalkLightGCN(graphmodule.LightGCNNet_dglnn):

    def forward(self, graphs, feat):
        if type(graphs) != list:
            return super().forward(graphs, feat)
        else:
            all_embeddings = [feat]
            for i in range(self.n_layers):
                graph = graphs[i]
                neigh_feat = self.conv_layer(i, graph, feat)
                feat = self.combiners[i](feat, neigh_feat)
                if self.normalize != None:
                    all_embeddings.append(F.normalize(feat, p=self.normalize))
                else:
                    all_embeddings.append(feat)
            return all_embeddings


r"""
SGL
#############
    SGL: Self-supervised Graph Learning for Recommendation (SIGIR'21)
    Reference:
        https://dl.acm.org/doi/10.1145/3404835.3462862
"""
class SGL(basemodel.BaseRetriever):

    def _init_model(self, train_data:TripletDataset):
        super()._init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.user_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        self.combiners = torch.nn.ModuleList()
        for i in range(self.config['model']['n_layers']):
            self.combiners.append(graphmodule.LightGCNCombiner(self.embed_dim, self.embed_dim))
        if self.config['model']['aug_type'] == 'RW':
            self.LightGCNNet = RandomWalkLightGCN(self.combiners)
        else:
            self.LightGCNNet = graphmodule.LightGCNNet_dglnn(self.combiners)
        adj_size = train_data.num_users + train_data.num_items
        self.adj_mat, _ = train_data.get_graph([0], form='dgl', value_fields='inter', \
            col_offset=[train_data.num_users], bidirectional=[True], shape=(adj_size, adj_size))
        # augmentation model
        self.augmentaion_model = data_augmentation.SGLAugmentation(self.config['model'], train_data)

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
        # [num_users + num_items, dim]
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        # {[num_users + num_items, dim], [num_users + num_items, dim], ..., [num_users + num_items, dim]}
        all_embeddings = self.LightGCNNet(self.adj_mat, embeddings)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        self.query_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        # TODO: make sure that padding embedding is all 0.

    def forward(self, batch, full_score, return_query=False, return_item=False, return_neg_item=False, return_neg_id=False):
        self.update_encoders()
        output = super().forward(batch, full_score, return_query=return_query, return_item=return_item, \
            return_neg_item=return_neg_item, return_neg_id=return_neg_id)
        return output

    def training_step(self, batch):
        model_config = self.config['model']
        output = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss), return_neg_id=True)
        cl_output = self.augmentaion_model(batch, self.user_emb, self.item_emb, self.adj_mat, self.LightGCNNet)
        l2_reg_loss = loss_func.l2_reg_loss_fn(
            self.user_emb(batch[self.fuid]),
            self.item_emb(batch[self.fiid]),
            self.item_emb(output['neg_id'].reshape(-1)))
        loss_value = self.loss_fn(batch[self.frating], **output['score']) \
            + model_config['l2_reg_weight'] * l2_reg_loss, \
            + model_config['ssl_reg'] * cl_output['cl_loss']
        return loss_value

    def _get_item_vector(self):
        if self.item_encoder.item_embeddings == None:
            return self.item_emb.weight[1:].detach().clone()
        else:
            return self.item_encoder.item_embeddings[1:].detach().clone()

    def _update_item_vector(self):
        self.update_encoders()
        super()._update_item_vector()