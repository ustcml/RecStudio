import torch
from torch import nn
import numpy as np
import math
from recstudio.data.dataset import AEDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
import copy
from recstudio.model.loss_func import SoftmaxLoss

def f(t, T, s=0.008):
    x = (t/T+s)/(1+s) * np.pi/2
    return np.cos(x) * np.cos(x)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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
    def __init__(self, num_items, embed_dim, dropout_rate) -> None:
        super(ScoreNet, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_items = num_items
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            SiLU(),
            nn.Linear(2*embed_dim, embed_dim),
        )
        self.encoders = torch.nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.Tanh(),
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.Tanh(),
            nn.Linear(2*embed_dim, 2*embed_dim),
        )
        self.t_layers = nn.ModuleList([nn.Sequential(nn.Tanh(), nn.Linear(embed_dim, 4*embed_dim)),
                                        nn.Sequential(nn.Tanh(), nn.Linear(4*embed_dim, 4*embed_dim)),
                                        nn.Sequential(nn.Tanh(), nn.Linear(4*embed_dim, 4*embed_dim))])
        self.cond_layers = nn.ModuleList([nn.Sequential(nn.Tanh(), nn.Linear(embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.Tanh(), nn.Linear(2*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.Tanh(), nn.Linear(2*embed_dim, 2*embed_dim))])
        self.in_layers = nn.ModuleList([nn.Sequential(nn.Linear(num_items, 2*embed_dim)),
                                        nn.Sequential(nn.Linear(2*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.Linear(2*embed_dim, 2*embed_dim))])
        self.out_layers = nn.ModuleList([nn.Sequential(nn.Dropout(p=self.dropout_rate), nn.Linear(2*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.Dropout(p=self.dropout_rate), nn.Linear(2*embed_dim, 2*embed_dim)),
                                        nn.Sequential(nn.Dropout(p=self.dropout_rate), nn.Linear(2*embed_dim, num_items))])
        

    def forward(self, x_t, t, x_start, training):
        x_t = 2*x_t - 1 #map to [-1,1]
        t_embed = self.time_embed(timestep_embedding(t, self.embed_dim))
        encoder_out = self.encoders(x_start)
        mu, logvar = encoder_out.tensor_split(2, dim=-1)
        cond_embed = self.reparameterize(mu, logvar, training)
        for i in range(3):
            t_embed = self.t_layers[i](t_embed)
            scale, shift = torch.chunk(t_embed, 2, dim=1)
            cond_embed = self.cond_layers[i](cond_embed)
            x_t = self.in_layers[i](x_t)
            x_t = x_t * (1+scale) + shift
            x_t = self.out_layers[i](x_t * cond_embed)
        kl_loss=None
        if training:
            kl_loss = self.kl_loss_func(mu, logvar) #torch.tensor(0.0).to(x_t.device)
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


class D3PMVAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, num_items, embed_dim, num_steps, beta_min, beta_max, margin, schedule, item_encoder, dropout_rate, kl_lambda):
        super().__init__()

        self.num_steps = num_steps
        self.margin = margin
        self.kl_lambda = kl_lambda
        if schedule==1:#linear:
            self.betas = torch.nn.Parameter(torch.linspace(beta_min, beta_max, self.num_steps), requires_grad=False)
        else:#cosine
            alphas = [f(t, num_steps)/f(0,num_steps) for t in range(num_steps+1)]
            betas = [1-alphas[i]/alphas[i-1] for i in range(1, num_steps+1)]
            self.betas = torch.nn.Parameter(torch.tensor(betas, requires_grad=False))
        self.Q_matrix = torch.nn.Parameter(torch.zeros(num_steps, 2, 2, dtype=torch.float32), requires_grad=False)
        for i in range(num_steps):
            self.Q_matrix[i,:,:] = torch.tensor([[1-0.5*self.betas[i], 0.5*self.betas[i]], [0.5*self.betas[i], 1-0.5*self.betas[i]]], dtype=torch.float32)
            #self.Q_matrix[i,:,:] = torch.tensor([[1.0, 0.0], [0.5*self.betas[i], 1-0.5*self.betas[i]]], dtype=torch.float32)
        self.Q_prod_matrix = copy.deepcopy(self.Q_matrix)
        for i in range(1, num_steps):
            self.Q_prod_matrix[i,:,:] = torch.matmul(self.Q_prod_matrix[i-1,:,:], self.Q_prod_matrix[i,:,:])

        self.dropout = nn.Dropout(p=dropout_rate)
        self.item_embedding = item_encoder
        self.fiid = fiid
        self.embed_dim = embed_dim
        self.num_items = num_items
        self.scorenet = ScoreNet(num_items-1, embed_dim, dropout_rate)
        self.loss_func = SoftmaxLoss()

    def forward(self, batch):
        seq_emb = self.item_embedding(batch["in_"+self.fiid])
        non_zero_num = batch["in_"+self.fiid].count_nonzero(dim=1).unsqueeze(-1)
        seq_emb = seq_emb.sum(1) / non_zero_num.pow(0.5)
        z = self.dropout(seq_emb)
        if self.training:
            output = self.diffusion_loss_func(batch["in_"+self.fiid], z)
        else:
            output = self.pc_sampler(batch["in_"+self.fiid], z)

        return output
    
    def diffusion_loss_func(self, item_ids, z):
        batch_size = item_ids.shape[0]

        x_int = torch.zeros([batch_size, self.num_items], dtype=torch.long, device=item_ids.device)
        x_int = x_int.scatter(1, item_ids, 1)
        x_int = x_int[:,1:] 
        x_one_hot  = torch.nn.functional.one_hot(x_int, num_classes=2).float()#(batch_size, num_items, 2)

        t = torch.randint(0, self.num_steps, size=(batch_size,), device=item_ids.device)

        Q_prod_t = self.Q_prod_matrix[t]#(batch_size, 2, 2)
        prob = torch.matmul(x_one_hot, Q_prod_t)#(batch_size, num_items, 2)
        x_perturbed = torch.distributions.Bernoulli(probs=prob[:,:,1]).sample()#(batch_size, num_items)

        output, kl_loss = self.scorenet(x_perturbed, t, z, True)
        #output += x_one_hot

        all_scores = output
        all_scores = torch.cat([-torch.inf * torch.ones([all_scores.shape[0], 1], device=all_scores.device), all_scores], dim=1)
        pos_scores = torch.gather(all_scores, 1, item_ids)
        loss = self.loss_func(None, pos_scores, all_scores)

        return loss + self.kl_lambda * kl_loss

    def pc_sampler(self, item_ids, z):
        with torch.no_grad():
            cur_x = torch.distributions.Bernoulli(probs=0.5*torch.ones([item_ids.shape[0], self.num_items], device=item_ids.device)).sample()
            #cur_x = torch.zeros([item_ids.shape[0], self.num_items], dtype=torch.float32, device=item_ids.device)
            cur_x = cur_x.scatter(1, item_ids, 1)
            cur_x = cur_x[:,1:]
            x0_1 = torch.nn.functional.one_hot(torch.ones_like(cur_x, dtype=torch.long, device=item_ids.device), num_classes=2).float()#(batch_size, num_items, 2)
            x0_0 = torch.nn.functional.one_hot(torch.zeros_like(cur_x, dtype=torch.long, device=item_ids.device), num_classes=2).float()
            for i in reversed(range(self.margin, self.num_steps, self.margin)):
                t = torch.tensor([i]*item_ids.shape[0], device=item_ids.device)
                x0_bar = self.scorenet(cur_x, t, z, False)[0] # (batch_size, num_items)
                cur_x_one_hot = torch.nn.functional.one_hot(cur_x.long(), num_classes=2).float()#(batch_size, num_items, 2)
                prob_1 = torch.matmul(cur_x_one_hot, self.Q_matrix[i].T) * torch.matmul(x0_1, self.Q_prod_matrix[i-self.margin]) \
                                / (torch.matmul(x0_1, self.Q_prod_matrix[i]) * cur_x_one_hot).sum(-1)[:,:,None]
                prob_0 = torch.matmul(cur_x_one_hot, self.Q_matrix[i].T) * torch.matmul(x0_0, self.Q_prod_matrix[i-self.margin]) \
                                / (torch.matmul(x0_0, self.Q_prod_matrix[i]) * cur_x_one_hot).sum(-1)[:,:,None]
                prob = x0_bar[:,:,None] * prob_1 + (1-x0_bar)[:,:,None] * prob_0
                cur_x = torch.argmax(prob, dim=-1).float()
                # prob_sample = torch.softmax(prob, dim=-1)
                # cur_x = torch.distributions.Bernoulli(probs=prob_sample[:, :, 1]).sample()
                #inpaint
                cur_x = torch.cat((torch.zeros([cur_x.shape[0], 1], dtype=torch.float32, device=cur_x.device), cur_x), dim=1)
                cur_x = cur_x.scatter(1, item_ids, 1)
                cur_x = cur_x[:,1:]
        return prob[:, :, 1]
            



class D3PMVAE(BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('D3PMVAE')
        parent_parser.add_argument("--num_steps", type=int, default=1000, help="total diffusion steps")
        parent_parser.add_argument("--beta_min", type=float, default=0.02, help="min value for beta")
        parent_parser.add_argument("--beta_max", type=float, default=1.0, help="max value for beta")
        parent_parser.add_argument("--margin", type=int, default=1, help="DDIM margin")
        parent_parser.add_argument("--schedule", type=int, default=1, help="noise schedule")
        parent_parser.add_argument("--dropout_rate", type=float, default=0.2, help="")
        parent_parser.add_argument("--kl_lambda", type=float, default=1.0, help="")
        return parent_parser

    def _get_dataset_class():
        return AEDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)

    def _get_query_encoder(self, train_data):
        return D3PMVAEQueryEncoder(train_data.fiid, train_data.num_items, self.embed_dim, self.config['num_steps'], self.config['beta_min'], self.config['beta_max'],
                                    self.config['margin'], self.config['schedule'], self.item_encoder, self.config['dropout_rate'], self.config['kl_lambda'])

    def _get_score_func(self):
        return None

    def _get_sampler(self, train_data):
        return None

    def _get_loss_func(self):
        return None

    def training_step(self, batch):
        loss_value = self.forward(batch)
        return loss_value

    def forward(self, batch):
        output = self.query_encoder(self._get_query_feat(batch))
        return output
    
    def topk(self, batch, k, user_h=None, return_query=False):
        # TODO: complete topk with retriever
        output = self.query_encoder(self._get_query_feat(batch))
        more = user_h.size(1) if user_h is not None else 0
        
        score, topk_items = torch.topk(output, k + more)
        topk_items = topk_items + 1
        if user_h is not None:
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)#idx of the first term in existing satisfies topk_items <= existing[idx],        shape:(batch_size, k+more)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score, idx = score.topk(k)
            topk_items = torch.gather(topk_items, 1, idx)

        if return_query:
            return score, topk_items, output
        else:
            return score, topk_items