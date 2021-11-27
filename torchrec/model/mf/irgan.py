from torchrec.data.advance_dataset import ALSDataset
from torchrec.model import basemodel, loss_func, scorer
import torch.nn.functional as F
from torch import optim
import torch
from torchrec.ann import sampler
import numpy as np
class IRGAN(basemodel.TwoTowerRecommender):

    def __init__(self, config):
        super().__init__(config)
        self.D = Discriminator(self.config)
        self.sample_lambda = 0.2
        self.T_gen = 1
        self.T_dis = 0.2
    
    def init_model(self, train_data):
        super().init_model(train_data)
        self.D.init_model(train_data)
        self.D.sampler = self.forward
    
    def init_parameter(self):
        super().init_parameter()

    def get_dataset_class(self):
        return ALSDataset

    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
    
    def config_scorer(self):
        return scorer.InnerProductScorer()
    
    def config_loss(self):
        pass

    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == self.iter_idx[self.current_epoch % sum(self.n_epochs)]:
            if optimizer_idx == 0:
                loss = self.D.training_step(batch, batch_idx)
                return {'loss': loss, 'd_loss': loss.detach()}
            if optimizer_idx == 1:
                loss = self._training_step(batch, self.D)
                return {'loss': loss, 'g_loss': loss.detach()}
        else:
            return None

    def _training_step(self, batch, discriminator):
        pos_items = self.get_item_feat(batch)
        query = self.construct_query(batch)
        weight, item_ids, prob_neg = self.forward(query, 2 * self.neg_count, pos_items, True)
        rewards = discriminator.reward_for_g(batch, item_ids, weight).detach()
        mask = (item_ids != 0).float()
        loss = -torch.mean(torch.sum(torch.log(prob_neg) * rewards, dim=1) / torch.sum(mask, dim=1))
        return loss

    def forward(self, query, num_neg, pos_items, isGen=False):
        logit = self.score_func(query, self.get_item_vector()) / (self.T_gen if isGen else self.T_dis)
        prob = F.pad(logit.softmax(dim=-1), (1, 0))
        if isGen:
            mask = (pos_items > 0).int()
            num_pos = mask.sum(dim=-1, keepdim=True)
            prob_with_imp_sampling = prob * (1 - self.sample_lambda)
            prob_with_imp_sampling.scatter_(1, pos_items, prob.gather(1, pos_items) + self.sample_lambda / num_pos)
            prob_with_imp_sampling[:, 0] = 0.0
            neg_items = torch.multinomial(prob_with_imp_sampling, num_neg * pos_items.size(-1), replacement=True) #* mask.repeat(1, num_neg)
            neg_prob = prob.gather(1, neg_items)
            weight= neg_prob / prob_with_imp_sampling.gather(1, neg_items)
            return weight.detach(), neg_items.detach(), neg_prob
        else:
            neg_items = torch.multinomial(prob, num_neg * pos_items.size(-1), replacement=True) #* mask.repeat(1, num_neg)
            neg_prob = prob.gather(1, neg_items)
            return prob.gather(1, pos_items).detach(), neg_items.detach(), neg_prob.detach()
    
    def configure_optimizers(self):
        opt_d = optim.Adam(self.D.parameters(), lr=0.001, weight_decay=self.config['regularization'])
        opt_g = optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return [opt_d, opt_g]

    def set_train_loaders(self, train_data):
        train_data.loaders = [train_data.loader, train_data.loader]
        train_data.nepoch = self.n_epochs = [5, 2]
        self.iter_idx = np.concatenate([np.repeat(i, c) for i, c in enumerate(self.n_epochs)])
        return False

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if optimizer_idx == self.iter_idx[epoch % sum(self.n_epochs)]:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer_closure()
    
    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return [callbacks[i] for i in range(1)]



class Discriminator(basemodel.TwoTowerRecommender):

    def build_user_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
    
    def config_loss(self):
        return loss_func.BinaryCrossEntropyLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()

    def build_sampler(self, train_data):
        return None

    def reward_for_g(self, batch, neg_item_idx, weight):
        query = self.construct_query(batch)
        neg_items = self.item_encoder(self.get_item_feat(neg_item_idx))
        reward = 2 * (torch.sigmoid(self.score_func(query, neg_items)) - 0.5) * weight 
        return reward
    
    
