from recstudio.model import basemodel, loss_func, scorer
from recstudio.ann import sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from recstudio.data import dataset
from copy import deepcopy

def swish(x):
    return x.mul(torch.sigmoid(x))

import numpy as np
def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp()) # check whether kldivloss in torch


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
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

class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
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
        
    def forward(self, batch_data, dropout):
        count_nonzero = batch_data.count_nonzero(dim=1).unsqueeze(-1) # batch_user * 1
        user_embs = self.encoder_layer_0(batch_data) # batch_user * dims
        user_embs = torch.sum(user_embs, dim=1) / count_nonzero.pow(0.5)
        h0 = F.dropout(user_embs, p=dropout)

        h1 = self.ln1(swish(h0))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)






class Rec_VAE(basemodel.ItemTowerRecommender):
    """
    RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback(WSDM'20)
    Reference:
    https://dl.acm.org/doi/abs/10.1145/3336191.3371831
    Github:
    https://github.com/ilya-shenbin/RecVAE
    """
    def __init__(self, config):
        config.update({"embed_dim":config['latent_dim']})
        super(Rec_VAE, self).__init__(config)

        # self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        self.dropout = config['dropout']
        self.gamma = config['gamma']
        self.alternating = config['alternating']
        if self.alternating:
            self.enc_n_epoch = config['enc_epoch']
            self.dec_n_epoch = config['dec_epoch']
            n_epochs = [config['enc_epoch'], config['dec_epoch'] ]
            self.iter_idx = np.concatenate([np.repeat(i, c) for i, c in enumerate(n_epochs)])

    def get_dataset_class(self):
        return dataset.AEDataset

    def init_model(self, train_data):
        super().init_model(train_data)
        self.input_dim = train_data.num_items
        self.encoder = Encoder(self.hidden_dim, self.latent_dim, self.input_dim)
        self.prior = CompositePrior(self.hidden_dim, self.latent_dim, self.input_dim)
        # self.decoder = nn.Linear(self.latent_dim, self.input_dim)
        # self.decoder = nn.Linear(self.latent_dim, self.hidden_dim)
    
    def config_loss(self):
        return loss_func.SoftmaxLoss()
    
    def config_scorer(self):
        return scorer.InnerProductScorer()

    def construct_query(self, batch_data):
        assert len(self.user_fields) == 1

        data = batch_data['in_item_id']
        # ratings = batch_data['in_rating']
        z, self.kld_loss = self._encoder_user(data, gamma=self.gamma)
        return z
    
    def _encoder_user(self, batch_data, beta=None, gamma=1,):
        mu, logvar = self.encoder(batch_data, self.dropout)
        z = self.reparameterize(mu, logvar)

        if gamma:
            norm = batch_data.count_nonzero(dim=1).unsqueeze(-1)
            kl_weight = gamma * norm
        elif beta:
            kl_weight = beta
        kld = (log_norm_pdf(z, mu, logvar) - self.prior(batch_data, z)).sum(dim=-1).mul(kl_weight).mean()
        return z, kld


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # optimizer_idx is required for multi optimizers
        if optimizer_idx == self.iter_idx[self.current_epoch % (self.enc_n_epoch + self.dec_n_epoch)]:
            loss = super().training_step(batch, batch_idx)
            return loss + self.kld_loss

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
    
    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        if self.alternating:
            if self.current_epoch % (self.enc_n_epoch + self.dec_n_epoch) == (self.enc_n_epoch - 1):
                self.update_prior()

    # Source code update the encoder and decoder alternately, control the two different optimizers
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if self.alternating:
            if optimizer_idx == self.iter_idx[epoch % (self.enc_n_epoch + self.dec_n_epoch)]:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
        else:
            if optimizer_idx in [0,1]:
                optimizer.step(closure=optimizer_closure)
            
    def configure_optimizers(self):
        opt_enc = self.get_optimizer(self.encoder.parameters())
        # Q: module CompositePrior seems to have no update of parameters
        opt_dec = self.get_optimizer(self.item_encoder.parameters())
        return [opt_enc, opt_dec]
        