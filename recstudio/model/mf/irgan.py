from recstudio.data.advance_dataset import ALSDataset
from recstudio.model import basemodel
from recstudio.ann import sampler
import torch.nn.functional as F
from torch import optim
import torch


# class IRGAN(basemodel.BaseRetriever):

#     def __init__(self, config):
#         super().__init__(config)
#         self.G = Generator(config)
#         self.D = Discriminator(config, retriever=self.G)
        
    
#     def _get_dataset_class(self):
#         return ALSDataset

    
#     def training_step(self, batch, nepoch):
#         epochs_per_cycle = self.config['every_n_epoch_gen'] + self.config['every_n_epoch_dis']
#         if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_dis']:
#             loss = self.D.training_step(batch)
#             return {'loss': loss, 'd_loss': loss.detach()}
#         else:
#             loss = self.G.training_step(batch, self.D)
#             return {'loss': loss, 'g_loss': loss.detach()}
    

#     def _get_optimizers(self):
#         opt_d = optim.Adam(self.D.parameters(), lr=self.config['learning_rate_dis'], weight_decay=self.config['weight_decay_dis'])
#         opt_g = optim.Adam(self.G.parameters(), lr=self.config['learning_rate_gen'], weight_decay=self.config['weight_decay_gen'])
#         return [{'optimizer': opt_d}, {'optimizer': opt_g}]


#     def current_epoch_optimizers(self, nepoch):
#         epochs_per_cycle = self.config['every_n_epoch_gen'] + self.config['every_n_epoch_dis']
#         if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_dis']:
#             return self.optimizers[:1]
#         else:
#             return self.optimizers[1:]

    
#     def test_epoch(self, batch):
#         return self.G.test_epoch(batch)
        

#     def validation_epoch(self, batch):
#         return self.G.validation_epoch(batch)


class IRGAN(basemodel.BaseRetriever):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.generator = Generator(config)

    def _init_model(self, train_data):
        self.generator._init_model(train_data)
        # self.retriever._init_parameter()
        super()._init_model(train_data)
    
    def _get_dataset_class(self):
        return ALSDataset


    def _get_loss_func(self):
        def bce_loss(label, pos_score, log_pos_prob, neg_score, log_neg_prob):
            mask = torch.logical_not(torch.isinf(pos_score))
            return torch.mean(torch.masked_select((-F.logsigmoid(pos_score) + F.softplus(neg_score)), mask))
        return bce_loss

    def _get_sampler(self, train_data):
        return sampler.RetriverSampler(self.generator, method='brute', excluding_hist=False, t=self.config['T_dis'])


    def _get_optimizers(self):
        opt_d = optim.Adam(self._dis_parameters(), lr=self.config['learning_rate_dis'], weight_decay=self.config['weight_decay_dis'])
        opt_g = optim.Adam(self.generator.parameters(), lr=self.config['learning_rate_gen'], weight_decay=self.config['weight_decay_gen'])
        return [{'optimizer': opt_d}, {'optimizer': opt_g}]


    def current_epoch_optimizers(self, nepoch):
        epochs_per_cycle = self.config['every_n_epoch_gen'] + self.config['every_n_epoch_dis']
        if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_dis']:
            return self.optimizers[:1]
        else:
            return self.optimizers[1:]


    def reward_for_g(self, batch, neg_item_idx, weight):
        with torch.no_grad():
            query = self.query_encoder(self._get_query_feat(batch))
            neg_items = self.item_encoder(self._get_item_feat(neg_item_idx))
            reward = 2 * (torch.sigmoid(self.score_func(query, neg_items)) - 0.5) * weight 
        return reward
    

    def _dis_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if ('sampler' not in name) and ('generator' not in name):
                yield param

    
    def training_step(self, batch, nepoch):
        epochs_per_cycle = self.config['every_n_epoch_gen'] + self.config['every_n_epoch_dis']
        if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_dis']:
            loss = super().training_step(batch)
            return {'loss': loss, 'd_loss': loss.detach()}
        else:
            loss = self.generator.training_step(batch, self)
            return {'loss': loss, 'g_loss': loss.detach()}

    
    def test_epoch(self, batch):
        return self.generator.test_epoch(batch)
        

    def validation_epoch(self, batch):
        return self.generator.validation_epoch(batch)



class Generator(basemodel.BaseRetriever):

    def _get_dataset_class(self):
        return None

    def _get_loss_func(self):
        return None
    
    def training_step(self, batch, discriminator):
        pos_items = self._get_item_feat(batch)
        query = self.query_encoder(self._get_query_feat(batch))
        weight, item_ids, prob_neg = self.forward(query, 2 * self.neg_count, pos_items)
        rewards = discriminator.reward_for_g(batch, item_ids, weight).detach()
        loss = -torch.sum(torch.mean(torch.log(prob_neg) * rewards, dim=1))
        return loss


    def forward(self, query, num_neg, pos_items):
        self._update_item_vector()
        logit = self.score_func(query, self.item_vector) / self.config['T_gen']
        prob = F.pad(logit.softmax(dim=-1), (1, 0))

        mask = (pos_items > 0).int()
        num_pos = mask.sum(dim=-1, keepdim=True)
        prob_with_imp_sampling = prob * (1 - self.config['sample_lambda'])
        prob_with_imp_sampling.scatter_(1, pos_items, prob_with_imp_sampling.gather(1, pos_items) + self.config['sample_lambda'] / num_pos)
        prob_with_imp_sampling[:, 0] = 0.0
        neg_items = torch.multinomial(prob_with_imp_sampling, num_neg * pos_items.size(-1), replacement=True) #* mask.repeat(1, num_neg)
        neg_prob = prob.gather(1, neg_items)
        weight= neg_prob / prob_with_imp_sampling.gather(1, neg_items)
        return weight.detach(), neg_items.detach(), neg_prob
