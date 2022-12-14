import torch
from torch import nn
import numpy as np
from recstudio.data.dataset import AEDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
from recstudio.model.module import MLPModule

class TimeEncoding(nn.Module):
    def __init__(self, embed_dim, scale=1.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim//2) * scale)#初始化可能被xavier_normal_覆盖了？
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNet(torch.nn.Module):
    def __init__(self, embed_dim, num_items, num_steps, dropout_rate, encoder_dims, activation, item_encoder) -> None:
        super(ScoreNet, self).__init__()
        self.encoders = torch.nn.Sequential(
            MLPModule([embed_dim]+encoder_dims[:-1], activation),
            torch.nn.Linear(([embed_dim]+encoder_dims[:-1])[-1], encoder_dims[-1])
        )

        self.embed_t = nn.Sequential(TimeEncoding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))#time encoding layer
        self.item_embedding = item_encoder

        self.linear1_x = torch.nn.Linear(num_items, 4*embed_dim)
        self.linear1_t = torch.nn.Linear(embed_dim, 4*embed_dim)
        self.linear1_u = torch.nn.Linear(embed_dim, 4*embed_dim)
        self.linear2_x = torch.nn.Linear(4*embed_dim, 4*embed_dim)
        self.linear2_t = torch.nn.Linear(embed_dim, 4*embed_dim)
        self.linear2_u = torch.nn.Linear(embed_dim, 4*embed_dim)
        # self.linear3_x = torch.nn.Linear(4*embed_dim, embed_dim)
        # self.linear3_t = torch.nn.Linear(embed_dim, embed_dim)
        # self.linear3_u = torch.nn.Linear(embed_dim, embed_dim)
        self.linear4_x = torch.nn.Linear(4*embed_dim, num_items)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.act = nn.Tanh()

    def forward(self, x, t, item_ids):
        embed_t = self.act(self.embed_t(t))
        user_emb = self.item_embedding(item_ids)
        non_zero_num = item_ids.count_nonzero(dim=1).unsqueeze(-1)
        user_emb = user_emb.sum(1) / non_zero_num.pow(0.5)
        user_emb = self.dropout(user_emb)
        user_emb = self.encoders(user_emb)

        x = self.act(self.linear1_x(x)) + self.act(self.linear1_t(embed_t)) + self.act(self.linear1_u(user_emb))
        x = self.act(self.linear2_x(x)) + self.act(self.linear2_t(embed_t)) + self.act(self.linear2_u(user_emb))
        #x = self.act(self.linear3_x(x)) + self.act(self.linear3_t(embed_t)) + self.act(self.linear3_u(user_emb))
        x = self.linear4_x(x)
        return x

class DDPMVAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, num_items, embed_dim, dropout_rate,
                 encoder_dims, activation, num_steps, beta_min, beta_max, item_encoder, mask_rate):
        super().__init__()

        self.num_steps = num_steps
        self.betas = torch.nn.Parameter(torch.linspace(beta_min, beta_max, self.num_steps), requires_grad=False)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_sqrt = torch.nn.Parameter(torch.sqrt(self.alphas_prod), requires_grad=False)
        self.one_minus_alphas_bar_sqrt = torch.nn.Parameter(torch.sqrt(1 - self.alphas_prod), requires_grad=False)

        self.fiid = fiid
        self.embed_dim = embed_dim
        self.num_items = num_items
        self.mask_rate = mask_rate
        self.scorenet = ScoreNet(embed_dim, num_items-1, self.num_steps, dropout_rate, encoder_dims, activation, item_encoder)

    def forward(self, batch):
        if self.training:
            output = self.diffusion_loss_func(batch["in_"+self.fiid])
        else:
            output = self.pc_sampler(batch["in_"+self.fiid])

        return output
    
    def diffusion_loss_func(self, item_ids):
        batch_size = item_ids.shape[0]

        rows = torch.arange(batch_size, device=item_ids.device).unsqueeze(-1).repeat(1, item_ids.shape[1]).reshape(-1)
        cols = item_ids.reshape(-1)
        v = torch.where(item_ids>0, torch.ones_like(item_ids, dtype=torch.int, device=item_ids.device), item_ids).reshape(-1)
        x = torch.sparse.IntTensor(torch.stack((rows, cols)), v, torch.Size([batch_size, self.num_items])).to_dense()  

        # probability_matrix = torch.full(x.shape, self.mask_rate, device=x.device)
        # mask_idx = torch.bernoulli(probability_matrix).bool()
        # x[0][0]=0.2
        # x[mask_idx] = 0.5

        t = torch.randint(0, self.num_steps, size=(batch_size,), device=item_ids.device)
        t = t.unsqueeze(-1)#(batch_size,1)

        a = self.alphas_bar_sqrt[t]
        am1 = self.one_minus_alphas_bar_sqrt[t]
        e = torch.randn([batch_size, self.num_items-1], device=item_ids.device)
        x = x[:,1:] * a + e * am1
        output = self.scorenet(x, t.squeeze(-1), item_ids)
        return (e-output).square().mean()

    def pc_sampler(self, item_ids):
        #t=torch.ones(user_emb.shape[0], device=user_emb.device)
        with torch.no_grad():
            cur_x = torch.randn(item_ids.shape[0], self.num_items-1, device=item_ids.device)
            for i in reversed(range(self.num_steps)):
                t = torch.tensor([i], device=item_ids.device)
                coeff = self.betas[t]/self.one_minus_alphas_bar_sqrt[t]
                eps_theta = self.scorenet(cur_x, t, item_ids)
                mean = (1/(1-self.betas[t]).sqrt()) * (cur_x - coeff * eps_theta)

                z = torch.randn_like(cur_x)
                sigma_t = self.betas[t].sqrt()
                cur_x = mean + sigma_t * z
        
        return -(mean-1)**2
            



class DDPMVAE(BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DDPMVAE')
        parent_parser.add_argument("--dropout", type=int, default=0.5, help='dropout rate for MLP layers')
        parent_parser.add_argument("--encoder_dims", type=int, nargs='+', default=64, help='MLP layer size for encoder')
        parent_parser.add_argument("--activation", type=str, default='relu', help='activation function for MLP layers')
        parent_parser.add_argument("--num_steps", type=int, default=100, help="total diffusion steps")
        parent_parser.add_argument("--beta_min", type=float, default=1e-4, help="min value for beta")
        parent_parser.add_argument("--beta_max", type=float, default=0.02, help="max value for beta")
        parent_parser.add_argument("--mask_rate", type=float, default=0.1, help="max value for beta")
        return parent_parser

    def _init_parameter(self):
        super()._init_parameter()
        # state_dict = torch.load('./saved/1stage_DiffusionVAE-ml-1m-2022-11-22-20-39-05.ckpt', map_location='cpu')
        # update_state_dict = {}
        # for key, value in state_dict['parameters'].items():
        #     if key=='item_encoder.weight' or key.startswith('query_encoder.encoders'):
        #         update_state_dict[key] = value
        # self.load_state_dict(update_state_dict, strict=False)

    def _get_dataset_class():
        return AEDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)

    def _get_query_encoder(self, train_data):
        return DDPMVAEQueryEncoder(train_data.fiid, train_data.num_items,
                                    self.embed_dim, self.config['dropout_rate'], self.config['encoder_dims'],self.config['activation'], 
                                    self.config['num_steps'], self.config['beta_min'], self.config['beta_max'], self.item_encoder, self.config['mask_rate'])

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
        output = self.query_encoder(self._get_query_feat(batch))#default dropout_rate=0
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