"""
Various utilities for neural networks.
"""

import math

import torch
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ScoreNet(nn.Module):
    def __init__(self, embed_dim, dropout_rate, learn_sigma) -> None:
        super(ScoreNet, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            SiLU(),
            nn.Linear(2*embed_dim, embed_dim),
        )
        self.encoders = torch.nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Linear(2*embed_dim, 2*embed_dim),
        )
        self.t_layers = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.ReLU(), nn.Linear(2*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.ReLU(), nn.Linear(2*embed_dim, 2*embed_dim))])
        self.cond_layers = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.ReLU(), nn.Linear(2*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.ReLU(), nn.Linear(2*embed_dim, 2*embed_dim))])
        self.in_layers = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.ReLU(), nn.Linear(2*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.ReLU(), nn.Linear(2*embed_dim, 2*embed_dim))])
        self.out_layers = nn.ModuleList([nn.Sequential(nn.Dropout(p=self.dropout_rate), nn.Linear(4*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.Dropout(p=self.dropout_rate), nn.Linear(4*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.Dropout(p=self.dropout_rate), nn.Linear(4*embed_dim, 2*embed_dim if learn_sigma else embed_dim))])
        

    def forward(self, x_t, t, x_start, training, **kwargs):
        t_embed = self.time_embed(timestep_embedding(t, self.embed_dim))
        encoder_out = self.encoders(x_start)
        mu, logvar = encoder_out.tensor_split(2, dim=-1)
        cond_embed = self.reparameterize(mu, logvar, training)
        for i in range(3):
            t_embed = self.t_layers[i](t_embed)
            cond_embed = self.cond_layers[i](cond_embed)
            x_t = self.in_layers[i](x_t)
            x_t = self.out_layers[i](torch.cat((x_t + t_embed, cond_embed), dim=-1))
        kl_loss=None
        if training:
            kl_loss = self.kl_loss_func(mu, logvar)
        return (x_t, kl_loss)

    def reparameterize(self, mu, logvar, training):
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def kl_loss_func(self, mu, logvar):
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return KLD
        