from torch.utils import data
from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
import torch
import torch.nn.functional as F
from torchrec.model.kg import KGLearning

class CKE_TransRTower(KGLearning.TransRTower):
    def config_loss(self):
        return loss_func.BPRLoss()

class CKE_item_encoder(torch.nn.Module):
    def __init__(self, item_emb, ent_emb):
        super().__init__()
        self.item_emb = item_emb
        self.ent_emb = ent_emb
    def forward(self, batch_data):
        return self.item_emb(batch_data) + self.ent_emb(batch_data)
        
r"""
CKE
#############
    Collaborative knowledge base embedding for recommender systems(KDD'16)
    Reference: 
        https://dl.acm.org/doi/10.1145/2939672.2939673
"""
class CKE(basemodel.TwoTowerRecommender):
    r"""
    CKE jointly learns the latent representations in collaborative filtering as well as items' semantic representations from the knowledge base, 
    using three elaborate components which can extract items' semantic representations from structural content, textual content and visual content, respectively. 
    In this implementation, we only use TransR to extract items' structural representations. 
    """
    def __init__(self, config):
        self.kg_index = config['kg_network_index'] # the knowledge graph index in the dataset configuration file.
        super().__init__(config)

    def init_model(self, train_data):
        self.item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        # kg tower
        self.TransRTower = CKE_TransRTower(self.config)
        self.TransRTower.init_model(train_data)
        super().init_model(train_data)

    def get_dataset_class(self):
        return dataset.MFDataset

    def set_train_loaders(self, train_data):
        # if iscombine == True, rec loader must be the first item of loaders.
        train_data.loaders = [train_data.loader, train_data.network_feat[self.kg_index].loader]
        train_data.nepoch = None
        return True

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def config_loss(self):
        return loss_func.BPRLoss()
    
    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def build_item_encoder(self, train_data):
        return CKE_item_encoder(self.item_emb, self.TransRTower.ent_emb)

    def training_step(self, batch, batch_idx):
        # combine the batches in kg and rec loaders.
        batch[0].update(batch[1])
        batch = batch[0]
        rec_score = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))
        kg_score = self.TransRTower.forward(batch)
        loss = self.loss_fn(batch[self.frating], *rec_score) + self.TransRTower.loss_fn(None, *kg_score)
        return loss

    def get_item_vector(self):
        return self.item_emb.weight[1 : ] + self.TransRTower.ent_emb.weight[1 : self.item_emb.num_embeddings]
