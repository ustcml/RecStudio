from recstudio.data.dataset import TripletDataset
from recstudio.model import basemodel, loss_func, scorer, module
from recstudio.model.module import graphmodule, data_augmentation
from recstudio.ann import sampler
import torch

r"""
NCL
#############
    NCL: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning (WWW'22)
    Reference:
        https://doi.org/10.1145/3485447.3512104
"""
class NCL(basemodel.BaseRetriever):

    def _init_model(self, train_data: TripletDataset):
        super()._init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.user_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        self.combiners = torch.nn.ModuleList()
        model_config = self.config['model']
        for i in range(max(model_config['n_layers'], model_config['hyper_layers'] * 2)):
            self.combiners.append(graphmodule.LightGCNCombiner(self.embed_dim, self.embed_dim))
        self.LightGCNNet = graphmodule.LightGCNNet_dglnn(self.combiners)

        adj_size = train_data.num_users + train_data.num_items
        self.adj_mat, _ = train_data.get_graph([0], form='dgl', value_fields='inter', \
            col_offset=[train_data.num_users], bidirectional=[True], shape=(adj_size, adj_size))

        # augmentation model
        self.augmentation_model = data_augmentation.NCLAugmentation(self.config['model'], train_data)

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
        center_embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        # {[num_users + num_items, dim], [num_users + num_items, dim], ..., [num_users + num_items, dim]}
        all_embeddings_list = self.LightGCNNet(self.adj_mat, center_embeddings)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.stack(all_embeddings_list, dim=-2)
        # [num_users + num_items, dim]
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        self.query_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return all_embeddings_list
        # TODO: make sure that padding embedding is all 0.

    def forward(self, batch, full_score, return_neg_id=False):
        all_embeddings_list = self.update_encoders()
        output = super().forward(batch, full_score, return_neg_id=return_neg_id)
        output['all_embeddings_list'] = all_embeddings_list
        return output

    def training_step(self, batch, nepoch):
        model_config = self.config['model']
        output = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss), True)
        aug_output = self.augmentation_model(batch, output['all_embeddings_list'])
        loss = self.loss_fn(batch[self.frating], **output['score']) + \
            model_config['l2_reg_weight'] * loss_func.l2_reg_loss_fn(self.user_emb(batch[self.fuid]), self.item_emb(batch[self.fiid]), \
                self.item_emb(output['neg_id'].reshape(-1))) + \
            model_config['ssl_reg'] * aug_output['structure_cl_loss']
        if nepoch >= self.config['train']['warm_up_epoch']:
            loss = loss + model_config['proto_reg'] * aug_output['semantic_cl_loss']
        return loss

    def _get_item_vector(self):
        if self.item_encoder.item_embeddings == None:
            return self.item_emb.weight[1:].detach().clone()
        else:
            return self.item_encoder.item_embeddings[1:].detach().clone()

    def _update_item_vector(self):
        self.update_encoders()
        super()._update_item_vector()

    def training_epoch(self, nepoch):
        if nepoch % self.config['train']['num_m_epoch'] == 0:
            self.logger.info('run e_step!')
            self.augmentation_model.e_step(self.user_emb, self.item_emb)
        return super().training_epoch(nepoch)