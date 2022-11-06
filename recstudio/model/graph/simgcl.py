from recstudio.data.dataset import MFDataset
from recstudio.model import basemodel, loss_func, scorer
from recstudio.model.module import graphmodule, data_augmentation
from recstudio.ann import sampler
import torch
import torch.nn.functional as F

class SimGCLNet(graphmodule.LightGCNNet_dglnn):

    def __init__(self, eps:torch.float, combiners:torch.nn.ModuleList, normalize:int=None, mess_norm:str='both') -> None:
        super().__init__(combiners, normalize, mess_norm)
        self.eps = eps 
    
    def forward(self, graph, feat, perturbed=False):
        all_embeddings = [] # skip the input embedding 
        for i in range(self.n_layers):
            neigh_feat = self.conv_layer(i, graph=graph, feat=feat)
            feat = self.combiners[i](feat, neigh_feat)
            if perturbed == True:
                random_noise = torch.rand_like(feat)
                feat = feat + torch.sign(feat) * F.normalize(random_noise, dim=-1) * self.eps
            if self.normalize != None:
                all_embeddings.append(F.normalize(feat, p=self.normalize))
            else:
                all_embeddings.append(feat)
        return all_embeddings

r"""
SimGCL
#############
    Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation (SIGIR'22)
    Reference: 
        https://doi.org/10.1145/3477495.3531937
"""
class SimGCL(basemodel.BaseRetriever):
 
    def _init_model(self, train_data:MFDataset):
        super()._init_model(train_data)
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.user_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        self.item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        self.combiners = torch.nn.ModuleList()
        for i in range(self.config['n_layers']):
            self.combiners.append(graphmodule.LightGCNCombiner(self.embed_dim, self.embed_dim))
        self.SimGCLNet = SimGCLNet(self.config['eps'], self.combiners)
        adj_size = train_data.num_users + train_data.num_items
        self.adj_mat, _ = train_data.get_graph([0], form='dgl', value_fields='inter', \
            col_offset=[train_data.num_users], bidirectional=[True], shape=(adj_size, adj_size))

        # contrastive learning
        self.augmentation_model = data_augmentation.SimGCLAugmentation(self.config, train_data)

    def _get_dataset_class():
        return MFDataset

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
        all_embeddings = self.SimGCLNet(self.adj_mat, embeddings)
        # [num_users + num_items, num_layers, dim]
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        # [num_users + num_items, dim]
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        self.query_encoder.user_embeddings, self.item_encoder.item_embeddings = \
             torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        # TODO: make sure that padding embedding is all 0.
  
    def forward(self, batch_data, full_score, return_neg_id=True):
        self.update_encoders()
        output = super().forward(batch_data, full_score, return_neg_id=return_neg_id)
        return output

    def training_step(self, batch):
        output = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss), True)
        cl_output = self.augmentation_model(batch, self.user_emb, self.item_emb, self.adj_mat, self.SimGCLNet)
        loss = self.loss_fn(batch[self.frating], **output['score']) + cl_output['cl_loss'] + \
            self.config['l2_reg_weight'] * loss_func.l2_reg_loss_fn(self.user_emb(batch[self.fuid]), self.item_emb(batch[self.fiid]), \
            self.item_emb(output['neg_id'].reshape(-1)))
        return loss 

    def _get_item_vector(self):
        if self.item_encoder.item_embeddings == None:
            return self.item_emb.weight[1:].detach().clone()
        else:
            return self.item_encoder.item_embeddings[1:].detach().clone()

    def _update_item_vector(self):
        self.update_encoders()
        super()._update_item_vector()
        
