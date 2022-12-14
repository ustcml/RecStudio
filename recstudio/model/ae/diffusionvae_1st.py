# import torch
# from torch import nn
# import numpy as np
# from recstudio.data.dataset import AEDataset
# from recstudio.model.basemodel import BaseRetriever, Recommender
# from recstudio.model.loss_func import SoftmaxLoss
# from recstudio.model.module import MLPModule
# from recstudio.model.scorer import InnerProductScorer
# import functools

# def marginal_prob_std(t, beta_max=20, beta_min=0.1):
#     """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
#     Args:    
#         t: A vector of time steps.
#         sigma: The $\sigma$ in our SDE.  
    
#     Returns:
#         The standard deviation.
#     """    
#     #任意t时刻的扰动后条件高斯分布的均值和标准差
#     mean = torch.exp(-(t**2)*(beta_max-beta_min)/4 - t*beta_min/2)
#     std  = torch.sqrt(1 - torch.exp(-(t**2)*(beta_max-beta_min)/2 - t*beta_min))
#     #torch.sqrt((sigma**(2 * t) - 1.0) / 2.0 / np.log(sigma))
#     return mean, std

# def diffusion_coeff(t, beta_max=20, beta_min=0.1):
#     """Compute the diffusion coefficient of our SDE.
#     Args:
#         t: A vector of time steps.
#         sigma: The $\sigma$ in our SDE.
    
#     Returns:
#         The vector of diffusion coefficients.
#     """
#     #计算任意t时刻的扩散系数，本例定义的SDE没有漂移系数
#     #sigma**t
#     return beta_min+t*(beta_max-beta_min) 

# class TimeEncoding(nn.Module):
#     def __init__(self, embed_dim, scale=10.) -> None:
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(embed_dim//2) * scale, requires_grad=False)
    
#     def forward(self, x):
#         x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
#         return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# class ScoreNet(torch.nn.Module):
#     def __init__(self, embed_dim, marginal_prob_std) -> None:
#         super(ScoreNet, self).__init__()
#         self.embed_t = nn.Sequential(TimeEncoding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))#time encoding layer

#         self.linear1_x = torch.nn.Linear(embed_dim, 2*embed_dim)
#         self.linear1_t = torch.nn.Linear(embed_dim, 2*embed_dim)
#         self.linear1_u = torch.nn.Linear(embed_dim, 2*embed_dim)
#         self.linear2_x = torch.nn.Linear(2*embed_dim, 2*embed_dim)
#         self.linear2_t = torch.nn.Linear(embed_dim, 2*embed_dim)
#         self.linear2_u = torch.nn.Linear(embed_dim, 2*embed_dim)
#         self.linear3_x = torch.nn.Linear(2*embed_dim, embed_dim)
#         self.linear3_t = torch.nn.Linear(embed_dim, embed_dim)
#         self.linear3_u = torch.nn.Linear(embed_dim, embed_dim)
#         self.act = nn.Tanh()
#         self.margin_prob_std = marginal_prob_std

#     def forward(self, x, t, user_emb):
#         embed_t = self.act(self.embed_t(t))

#         x = self.act(self.linear1_x(x) + self.linear1_t(embed_t) + self.linear1_u(user_emb))
#         x = self.act(self.linear2_x(x) + self.linear2_t(embed_t) + self.linear2_u(user_emb))
#         x = self.linear3_x(x) + self.linear3_t(embed_t) + self.linear3_u(user_emb)
#         #x = x / self.margin_prob_std(t)[:, None]
#         return x

# class DiffusionVAEQueryEncoder(torch.nn.Module):
#     def __init__(self, fiid, num_items, embed_dim, dropout_rate,
#                  encoder_dims, item_encoder, share_item_encoder, activation):
#         super().__init__()

#         self.fiid = fiid
#         self.embed_dim = embed_dim
#         self.item_embedding = item_encoder if share_item_encoder==1 else torch.nn.Embedding(num_items, embed_dim, 0)
#         self.dropout = torch.nn.Dropout(p=dropout_rate)
#         self.encoders = torch.nn.Sequential(
#             MLPModule([embed_dim]+encoder_dims[:-1], activation),
#             torch.nn.Linear(([embed_dim]+encoder_dims[:-1])[-1], encoder_dims[-1]*2)
#         )
#         self.kl_loss = 0.0
#         self.diffusion_loss = 0.0
#         self.marginal_prob_std = functools.partial(marginal_prob_std)
#         self.diffusion_coeff = functools.partial(diffusion_coeff)
#         self.scorenet = ScoreNet(embed_dim, marginal_prob_std=self.marginal_prob_std)
        

#     def forward(self, batch):
#         # encode
#         seq_emb = self.item_embedding(batch["in_"+self.fiid])
#         non_zero_num = batch["in_"+self.fiid].count_nonzero(dim=1).unsqueeze(-1)
#         seq_emb = seq_emb.sum(1) / non_zero_num.pow(0.5)

#         if self.training:
#             h = self.dropout(seq_emb)
#             encoder_h = self.encoders(h)
#             mu, logvar = encoder_h.tensor_split(2, dim=-1)
#             z = self.reparameterize(mu, logvar)

#             self.kl_loss = self.kl_loss_func(mu, logvar)

#             self.diffusion_loss = self.diffusion_loss_func(z, seq_emb)
#         else:
#             z = self.pc_sampler(seq_emb)

#         return z

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def kl_loss_func(self, mu, logvar):
#         KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
#         return KLD
    
#     def diffusion_loss_func(self, x, user_emb, eps=1e-5):
#         random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps

#         z = torch.randn_like(x)
#         mean, std = self.marginal_prob_std(random_t)
#         perturbed_x = mean[:, None] * x + z * std[:, None]

#         score = self.scorenet(perturbed_x, random_t, user_emb)

#         loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
#         return loss

#     def pc_sampler(self, user_emb, num_steps=500, snr=0.16, eps=1e-3):
#         #t=torch.ones(user_emb.shape[0], device=user_emb.device)
#         init_x = torch.randn(user_emb.shape[0], self.embed_dim, device=user_emb.device)

#         #定义采样的逆时间网络以及每一步的时间步长
#         time_steps = np.linspace(1., eps, num_steps)#对连续时间的采样
#         step_size = time_steps[0] - time_steps[1]

#         x = init_x
#         with torch.no_grad():
#             for time_step in time_steps:
#                 batch_time_step = torch.ones(user_emb.shape[0], device=user_emb.device) * time_step

#                 #Corrector step(Langevin MCMC)
#                 grad = self.scorenet(x, batch_time_step, user_emb)
#                 grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
#                 noise_norm = np.sqrt(np.prod(x.shape[1:]))
#                 langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2#调整步长是为了保证信噪比固定
#                 #print(f"langevin_step_size: {langevin_step_size}")

#                 for _ in range(10):
#                     x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)#郎之万采样的迭代公式
#                     grad = self.scorenet(x, batch_time_step, user_emb)
#                     grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
#                     noise_norm = np.sqrt(np.prod(x.shape[1:]))
#                     langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
#                     #print(f"langevin_step_size: {langevin_step_size}")
                
#                 #Predictor step(Euler-Maruyama)
#                 g = self.diffusion_coeff(batch_time_step)
#                 x_mean = x + (g[:,None] * x / 2 + g[:, None] * self.scorenet(x, batch_time_step, user_emb)) * step_size
#                 x = x_mean + torch.sqrt(g * step_size)[:, None] * torch.randn_like(x)

#         return x_mean


# class DiffusionVAE(BaseRetriever):

#     def add_model_specific_args(parent_parser):
#         parent_parser = Recommender.add_model_specific_args(parent_parser)
#         parent_parser.add_argument_group('DiffusionVAE')
#         parent_parser.add_argument("--dropout", type=int, default=0.5, help='dropout rate for MLP layers')
#         parent_parser.add_argument("--encoder_dims", type=int, nargs='+', default=64, help='MLP layer size for encoder')
#         #parent_parser.add_argument("--decoder_dims", type=int, nargs='+', default=64, help='MLP layer size for decocer')
#         parent_parser.add_argument("--activation", type=str, default='relu', help='activation function for MLP layers')
#         parent_parser.add_argument("--anneal_max", type=float, default=1.0, help="max anneal coef for KL loss")
#         parent_parser.add_argument("--anneal_total_step", type=int, default=2000, help="total anneal steps")
#         parent_parser.add_argument("--share_item_encoder", type=int, default=0, help="")
#         #parent_parser.add_argument("--sigma", type=float, default=25.0, help="")
#         return parent_parser

#     def _get_dataset_class():
#         return AEDataset

#     def _get_item_encoder(self, train_data):
#         return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)#pad for 0

#     def _get_query_encoder(self, train_data):
#         return DiffusionVAEQueryEncoder(train_data.fiid, train_data.num_items,
#                                     self.embed_dim, self.config['dropout_rate'], self.config['encoder_dims'],
#                                     self.item_encoder, self.config['share_item_encoder'], self.config['activation'])#, self.config['sigma'])

#     def _get_score_func(self):
#         return InnerProductScorer()

#     def _get_sampler(self, train_data):
#         return None

#     def _get_loss_func(self):
#         self.anneal = 0.0
#         return SoftmaxLoss()
#     def training_step(self, batch):
#         loss = super().training_step(batch)
#         anneal = min(self.config['anneal_max'], self.anneal)
#         self.anneal = min(self.config['anneal_max'],
#                           self.anneal + (1.0 / self.config['anneal_total_step']))
#         print(f"loss : {loss}, kl_loss: {self.query_encoder.kl_loss}, diffusion_loss: {self.query_encoder.diffusion_loss}")
#         return loss + self.query_encoder.kl_loss + 0.1*self.query_encoder.diffusion_loss
