from numpy.core.fromnumeric import std
from torchrec.model.basemodel import UserItemIDTowerRecommender
from torchrec.model import loss_func, scorer
from torchrec.data.advance_dataset import ALSDataset
from torch.nn.init import normal_, constant_
import torch
class WRMF(UserItemIDTowerRecommender):

    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False

    def get_dataset_class(self):
        return ALSDataset
    
    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
    
    def on_train_epoch_start(self) -> None:
        self.PtP = self.user_encoder.weight.T @ self.user_encoder.weight
        self.QtQ = self.item_encoder.weight.T @ self.item_encoder.weight
    
    def init_parameter(self):
        super().init_parameter()
        self.user_encoder.weight.requires_grad=False
        self.item_encoder.weight.requires_grad=False
    
    def config_loss(self):
        return None
    
    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def training_step(self, batch, batch_idx):
        eye = self.config['lambda'] * torch.eye(self.embed_dim)
        if batch[self.fuid].dim() == 1: ## user model, updating user embedding
            item_embed = self.item_encoder(self.get_item_feat(batch)) # B x N x D
            QuQ = torch.bmm(item_embed.transpose(1, 2), item_embed) * self.config['alpha'] + self.QtQ + eye # BxDxD
            r = torch.bmm(item_embed.transpose(1, 2), batch[self.frating].unsqueeze(-1)).squeeze(-1) # BxD
            output = torch.linalg.solve(QuQ, r * (self.config['alpha'] + 1))
            self.user_encoder.weight[batch[self.fuid]] = output # B x D
        else:
            user_embed = self.user_encoder(self.get_user_feat(batch))
            PiP = torch.bmm(user_embed.transpose(1, 2), user_embed) * self.config['alpha'] + self.PtP + eye
            r = torch.bmm(user_embed.transpose(1, 2), batch[self.frating].float().unsqueeze(-1)).squeeze(-1)
            output = torch.linalg.solve(PiP, r * (self.config['alpha'] + 1))
            self.item_encoder.weight[batch[self.fiid]] = output
        
        return {'loss':torch.tensor(0.0)}