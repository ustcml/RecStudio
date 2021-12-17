from torch.utils import data
from torchrec.model import basemodel, loss_func, scorer
from torchrec.data import dataset
from torchrec.ann import sampler 
import torch
import torch.nn.functional as F

class KGbased_item_encoder(torch.nn.Module):
    def __init__(self, rec_item_emb, ent_emb):
        super().__init__()
        self.rec_item_emb = rec_item_emb
        self.ent_emb = ent_emb
    def forward(self, batch_data):
        return self.rec_item_emb(batch_data) + self.ent_emb(batch_data)

class CKE(basemodel.TwoTowerRecommender):
    def init_model(self, train_data):
        self.ent_emb = torch.nn.Embedding(train_data.num_entities, self.embed_dim, padding_idx=0)
        self.rec_item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        super().init_model(train_data)
        # kg embeddings 
        self.pro_embed_dim = self.config['pro_embed_dim']
        self.fhid = train_data.fhid
        self.ftid = train_data.ftid
        self.frid = train_data.frid
        self.pro_matrix_emb = torch.nn.Embedding(train_data.num_relations, self.embed_dim * self.pro_embed_dim, padding_idx=0)
        self.rel_emb = torch.nn.Embedding(train_data.num_relations, self.pro_embed_dim, padding_idx=0)
        
        # kg sampler and loss 
        self.kg_sampler = sampler.UniformSampler(train_data.num_entities - 1)
        self.kg_loss_fn = loss_func.BPRLoss()

    def get_dataset_class(self):
        return dataset.KnowledgeBasedDataset

    def set_train_loaders(self, train_data):
        return False

    def config_scorer(self):
        return scorer.InnerProductScorer()

    def config_loss(self):
        return loss_func.BPRLoss()
    
    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def build_item_encoder(self, train_data):
        return KGbased_item_encoder(self.rec_item_emb, self.ent_emb)

    def training_step(self, batch, batch_idx):
        y_h = self.forward(batch, isinstance(self.loss_fn, loss_func.FullScoreLoss))

        h_e = self.item_encoder.ent_emb(batch[self.fhid]).unsqueeze(1) # [batch_size, 1, dim] 
        pos_t_e = self.item_encoder.ent_emb(batch[self.ftid]).unsqueeze(1)
        r_e = self.rel_emb(batch[self.frid]) # [batch_size, pro_embed_dim]
        pro_e = self.pro_matrix_emb(batch[self.frid]).reshape(-1, self.embed_dim, self.pro_embed_dim) # [batch_size, dim, pro_dim]
        
        pos_prob, neg_t_id, neg_prob = self.kg_sampler(h_e.squeeze(1), self.neg_count, batch[self.ftid]) # neg_id : [batch_size, 1]
        neg_t_e = self.ent_emb(neg_t_id)
        
        h_e = (torch.bmm(h_e, pro_e)).squeeze(1) # [batch_size, pro_dim]
        pos_t_e = torch.bmm(pos_t_e, pro_e).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, pro_e).squeeze(1)
        r_e = F.normalize(r_e, p=2, dim=1)
        h_e = F.normalize(h_e, p=2, dim=1)
        pos_t_e = F.normalize(pos_t_e, p=2, dim=1)
        neg_t_e = F.normalize(neg_t_e, p=2, dim=1)
        pos_kg_score = -((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_kg_score = -((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_y_h = (pos_kg_score, pos_prob, neg_kg_score.squeeze(-1), neg_prob)

        loss = self.loss_fn(batch[self.frating], *y_h) + self.kg_loss_fn(None, *kg_y_h)
        return loss

    def get_item_vector(self):
        return self.rec_item_emb.weight[1 : ] + self.ent_emb.weight[1 : self.rec_item_emb.num_embeddings]
