from numpy.core.fromnumeric import std
from torchrec.model.basemodel import TwoTowerRecommender
from torchrec.model import loss_func, scorer
from torchrec.data.advance_dataset import ALSDataset
from torch.nn.init import normal_, constant_
import torch
class WRMF(TwoTowerRecommender):

    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False
    
    def set_train_loaders(self, train_data):
        train_data.loaders = [train_data.loader, train_data.transpose().loader]
        train_data.nepoch = None
        return False ## use combine loader or concate loaders

    def get_dataset_class(self):
        return ALSDataset
    
    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.item_encoder.embedding_dim, padding_idx=0)
    
    def on_train_epoch_start(self) -> None:
        self.PtP = self.user_encoder.weight.T @ self.user_encoder.weight
        self.QtQ = self.item_encoder.weight.T @ self.item_encoder.weight
    
    def init_parameter(self):
        super().init_parameter()
        self.user_encoder.weight.requires_grad=False
        self.item_encoder.weight.requires_grad=False
        self.register_buffer('eye', self.config['lambda'] * torch.eye(self.item_encoder.embedding_dim))
    
    def config_loss(self):
        return None
    
    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def construct_query(self, batch_data):
        return self.user_encoder(self.get_user_feat(batch_data))
    
    def training_step(self, batch, batch_idx):
        ratings = (batch[self.frating]>0).float()
        if batch[self.fuid].dim() == 1: ## user model, updating user embedding
            item_embed = self.item_encoder(self.get_item_feat(batch)) # B x N x D
            QuQ = torch.bmm(item_embed.transpose(1, 2), item_embed) * self.config['alpha'] + (self.QtQ + self.eye) # BxDxD
            r = torch.bmm(item_embed.transpose(1, 2), ratings.unsqueeze(-1)).squeeze(-1) # BxD
            output = torch.linalg.solve(QuQ, r * (self.config['alpha'] + 1))
            if self.config['item_bias']:
                output[:, -1] = 1.0
            self.user_encoder.weight[batch[self.fuid]] = output # B x D
            user_embed = self.user_encoder(self.get_user_feat(batch)) # BxD
            pred = self.score_func(user_embed, item_embed) # BxD   BxNxD -> BxN
            reg1 = torch.multiply(user_embed @ self.QtQ,  user_embed).sum(-1)
        else:
            user_embed = self.user_encoder(self.get_user_feat(batch))
            PiP = torch.bmm(user_embed.transpose(1, 2), user_embed) * self.config['alpha'] + (self.PtP + self.eye)
            r = torch.bmm(user_embed.transpose(1, 2), ratings.unsqueeze(-1)).squeeze(-1)
            output = torch.linalg.solve(PiP, r * (self.config['alpha'] + 1))
            self.item_encoder.weight[batch[self.fiid]] = output
            item_embed = self.item_encoder(self.get_item_feat(batch)) 
            pred = self.score_func(item_embed, user_embed) # BxD BxNxD  -> BxN
            reg1 = torch.multiply(item_embed @ self.PtP, item_embed).sum(-1)
        loss = torch.sum((ratings - pred) **2, dim=-1) * (self.config['alpha'] + 1)
        loss -= (pred**2).sum(-1)
        loss += reg1
        
        return {'loss':loss}