import torch
import torch.nn as nn
import torch.nn.functional as F
from recstudio.data.dataset import AEDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
from recstudio.model.loss_func import SoftmaxLoss
from recstudio.model.scorer import InnerProductScorer
import numpy as np
from typing import Dict, List
from copy import deepcopy
from ..loss_func import FullScoreLoss


def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):#calculate log(PDF(x)) for  N(mu,var)
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

class CompositePrior(nn.Module):
    def __init__(self, fiid, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = RecVAEQueryEncoder(fiid, hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

class RecVAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, hidden_dim, latent_dim, input_dim, eps=1e-1):#input_dim==num items
        super().__init__()

        self.fiid = fiid
        self.encoder_layer_0 = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)           


    def forward(self, batch, dropout_rate=0):
        user_emb = self.encoder_layer_0(batch["in_"+self.fiid])
        non_zero_num = batch["in_"+self.fiid].count_nonzero(dim=1).unsqueeze(-1)
        user_emb = user_emb.sum(1) / non_zero_num.pow(0.5)

        h0 = F.dropout(user_emb, p=dropout_rate, training=self.training)
        h1 = self.ln1(swish(h0))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        if self.training:
            return self.fc_mu(h5), self.fc_logvar(h5)
        else:
            return self.fc_mu(h5)


class RecVAE(BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('RecVAE')
        parent_parser.add_argument("--dropout", type=int, default=0.5, help='dropout rate for MLP layers')
        parent_parser.add_argument("--hidden_dim", type=int, default=64, help='')
        parent_parser.add_argument("--latent_dim", type=int, default=64, help='')
        parent_parser.add_argument("--gamma", type=float, default=0.00005, help="")
        parent_parser.add_argument("--beta", type=float, default=0.00005, help="")
        parent_parser.add_argument("--alternating", type=bool, default=True, help="")
        parent_parser.add_argument("--enc_epoch", type=int, default=3, help='')
        parent_parser.add_argument("--dec_epoch", type=int, default=1, help='')

        return parent_parser
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.prior = CompositePrior(train_data.fiid, self.config['hidden_dim'],self.config['latent_dim'],train_data.num_items)
        self.dropout_rate = self.config['dropout']
        self.alternating = self.config['alternating']
        self.gamma = self.config['gamma']
        self.beta = self.config['beta']
        if self.alternating:
            self.enc_n_epoch = self.config['enc_epoch']
            self.dec_n_epoch = self.config['dec_epoch']


    def _get_dataset_class():
        return AEDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.config['latent_dim'], 0)#0是pad

    def _get_query_encoder(self, train_data):
        return RecVAEQueryEncoder(train_data.fiid, self.config['hidden_dim'],self.config['latent_dim'],train_data.num_items)

    def _get_score_func(self):
        return InnerProductScorer()

    def _get_sampler(self, train_data):
        return None

    def _get_loss_func(self):
        return SoftmaxLoss()
    
    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.query_encoder.state_dict()))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def training_step(self, batch, nepoch):
        self.nepoch = nepoch
        dropout = 0 if nepoch % (self.enc_n_epoch + self.dec_n_epoch)>=self.enc_n_epoch and self.alternating else self.dropout_rate
        output = self.forward(batch, isinstance(self.loss_fn, FullScoreLoss), dropout)
        score = output['score']
        score['label'] = batch[self.frating]
        loss = self.loss_fn(**score)#construction error

        return loss + self.kl_loss

    def forward(self, batch, full_score, dropout_rate, return_query=False, return_item=False, return_neg_item=False, return_neg_id=False):
        # query_vec, pos_item_vec, neg_item_vec,
        output = {}
        pos_items = self._get_item_feat(batch)
        pos_item_vec = self.item_encoder(pos_items)
        if self.sampler is not None:
            pass
        else:
            batch_data=self._get_query_feat(batch)
            mu, logvar = self.query_encoder(batch_data, dropout_rate)
            query = self.reparameterize(mu, logvar)

            pos_score = self.score_func(query, pos_item_vec)
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf')  # padding
            output['score'] = {'pos_score': pos_score}
            if full_score:
                item_vectors = self._get_item_vector()
                all_item_scores = self.score_func(query, item_vectors)
                output['score']['all_score'] = all_item_scores

            if self.gamma:
                norm = batch_data["in_"+self.fiid].count_nonzero(dim=1).unsqueeze(-1)
                kl_weight = self.gamma * norm
            elif self.beta:
                kl_weight = self.beta
            self.kl_loss = (log_norm_pdf(query, mu, logvar) - self.prior(batch_data, query)).sum(dim=-1).mul(kl_weight).mean()

        if return_query:
            output['query'] = query
        if return_item:
            output['item'] = pos_item_vec
        return output

    def training_epoch_end(self, output_list):
        super().training_epoch_end(output_list)
        if self.nepoch % (self.enc_n_epoch + self.dec_n_epoch)==self.enc_n_epoch-1:
            self.update_prior()

    def _get_optimizers(self) -> List[Dict]:
        if self.alternating:
            params_encoder = self.query_encoder.parameters()
            params_decoder = self.item_encoder.parameters()
            
            optimizer_encoder = self._get_optimizer(self.config['learner'], params_encoder, self.config['learning_rate'], self.config['weight_decay'])
            scheduler_encoder = self._get_scheduler(self.config['scheduler'], optimizer_encoder)
            optimizer_decoder = self._get_optimizer(self.config['learner'], params_decoder, self.config['learning_rate'], self.config['weight_decay'])
            scheduler_decoder = self._get_scheduler(self.config['scheduler'], optimizer_decoder)
            m = self.val_metric if self.val_check else 'train_loss'
            if scheduler_encoder and scheduler_decoder:
                return [{
                    'optimizer': optimizer_encoder,
                    'lr_scheduler': {
                        'scheduler': scheduler_encoder,
                        'monitor': m,
                        'interval': 'epoch',
                        'frequency': 1,
                        'strict': False
                        }
                    },
                    {
                    'optimizer': optimizer_decoder,
                    'lr_scheduler': {
                        'scheduler': scheduler_decoder,
                        'monitor': m,
                        'interval': 'epoch',
                        'frequency': 1,
                        'strict': False
                    }
                }]
            else:
                return [{'optimizer': optimizer_encoder},{'optimizer':optimizer_decoder}]
        else:
            return super()._get_optimizers()

    def current_epoch_optimizers(self, nepoch) -> List:
        # use nepoch to config current optimizers
        if self.alternating:
            idx = 1 if nepoch % (self.enc_n_epoch + self.dec_n_epoch)>=self.enc_n_epoch else 0
            return self.optimizers[idx]
        return self.optimizers

   
