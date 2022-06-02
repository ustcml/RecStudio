from email.mime import base

from yaml import load
from recstudio.model import basemodel, loss_func, scorer
from recstudio.data.advance_dataset import ALSDataset
import torch
class WRMF(basemodel.BaseRetriever):

    def _get_dataset_class(self):
        return ALSDataset
    
    
    def _get_train_loaders(self, train_data):
        loader = train_data.loader(
            batch_size = self.config['batch_size'],
            shuffle = True, 
            num_workers = self.config['num_workers'], 
            drop_last = False)
        loader_T = train_data.transpose().loader(
            batch_size = self.config['batch_size'],
            shuffle = True, 
            num_workers = self.config['num_workers'], 
            drop_last = False)
        return [loader, loader_T] ## use combine loader or concate loaders


    def current_epoch_trainloaders(self, nepoch):
        return self.trainloaders[nepoch % len(self.trainloaders)], False
    
    def _get_optimizers(self):
        return None


    def training_epoch(self, nepoch):
        self.PtP = self.query_encoder.weight.T @ self.query_encoder.weight
        self.QtQ = self.item_encoder.weight.T @ self.item_encoder.weight
        return super().training_epoch(nepoch)


    def _init_parameter(self):
        super()._init_parameter()
        self.query_encoder.weight.requires_grad=False
        self.item_encoder.weight.requires_grad=False
        self.register_buffer('eye', self.config['lambda'] * torch.eye(self.embed_dim))
    
    
    def _get_loss_func(self):
        return None
    

    def _get_score_func(self):
        return scorer.InnerProductScorer()
    
    
    def training_step(self, batch):
        ratings = (batch[self.frating]>0).float()
        if batch[self.fuid].dim() == 1: ## user model, updating user embedding
            item_embed = self.item_encoder(self._get_item_feat(batch)) # B x N x D
            QuQ = torch.bmm(item_embed.transpose(1, 2), item_embed) * self.config['alpha'] + (self.QtQ + self.eye) # BxDxD
            r = torch.bmm(item_embed.transpose(1, 2), ratings.unsqueeze(-1)).squeeze(-1) # BxD
            output = torch.linalg.solve(QuQ, r * (self.config['alpha'] + 1))
            if self.config['item_bias']:
                output[:, -1] = 1.0
            self.query_encoder.weight[batch[self.fuid]] = output # B x D
            user_embed = self.query_encoder(self._get_query_feat(batch)) # BxD
            pred = self.score_func(user_embed, item_embed) # BxD   BxNxD -> BxN
            reg1 = torch.multiply(user_embed @ self.QtQ,  user_embed).sum(-1)
        else:
            user_embed = self.query_encoder(self._get_query_feat(batch))
            PiP = torch.bmm(user_embed.transpose(1, 2), user_embed) * self.config['alpha'] + (self.PtP + self.eye)
            r = torch.bmm(user_embed.transpose(1, 2), ratings.unsqueeze(-1)).squeeze(-1)
            output = torch.linalg.solve(PiP, r * (self.config['alpha'] + 1))
            self.item_encoder.weight[batch[self.fiid]] = output
            item_embed = self.item_encoder(self._get_item_feat(batch))
            pred = self.score_func(item_embed, user_embed) # BxD BxNxD  -> BxN
            reg1 = torch.multiply(item_embed @ self.PtP, item_embed).sum(-1)
        loss = torch.sum((ratings - pred) **2, dim=-1) * (self.config['alpha'] + 1)
        loss -= (pred**2).sum(-1)
        loss += reg1
        
        return {'loss':loss}