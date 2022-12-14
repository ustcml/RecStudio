import torch
from torch import nn
import numpy as np
from recstudio.data.dataset import AEDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
import copy

def f(t, T, s=0.008):
    x = (t/T+s)/(1+s) * np.pi/2
    return np.cos(x) * np.cos(x)

class TimeEncoding(nn.Module):
    def __init__(self, embed_dim, scale=1.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim//2) * scale)#初始化可能被xavier_normal_覆盖了？
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNet(torch.nn.Module):
    def __init__(self, embed_dim, num_items) -> None:
        super(ScoreNet, self).__init__()

        self.embed_t = nn.Sequential(TimeEncoding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))#time encoding layer

        self.linear1_x = torch.nn.Linear(num_items, 4*embed_dim)
        self.linear1_t = torch.nn.Linear(embed_dim, 4*embed_dim)
        self.linear2_x = torch.nn.Linear(4*embed_dim, 4*embed_dim)
        self.linear2_t = torch.nn.Linear(embed_dim, 4*embed_dim)
        self.linear3_x = torch.nn.Linear(4*embed_dim, 4*embed_dim)
        self.linear3_t = torch.nn.Linear(embed_dim, 4*embed_dim)
        self.linear4_x = torch.nn.Linear(4*embed_dim, num_items)
        self.act = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):#x is x_int
        x = 2*x - 1 #map to [-1,1]
        embed_t = self.act(self.embed_t(t))
        x = self.act(self.linear1_x(x) + self.linear1_t(embed_t))
        x = self.act(self.linear2_x(x) + self.linear2_t(embed_t))
        x = self.act(self.linear3_x(x) + self.linear3_t(embed_t))
        x = self.sigmoid(self.linear4_x(x))
        return x #torch.cat([(1.-x).unsqueeze(-1), x.unsqueeze(-1)], dim=-1)

class D3PMVAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, num_items, embed_dim, num_steps, beta_min, beta_max, alpha, margin, schedule):
        super().__init__()

        self.num_steps = num_steps
        self.alpha = alpha
        self.margin = margin
        if schedule==1:#linear:
            self.betas = torch.nn.Parameter(torch.linspace(beta_min, beta_max, self.num_steps), requires_grad=False)
        else:#cosine
            alphas = [f(t, num_steps)/f(0,num_steps) for t in range(num_steps+1)]
            betas = [1-alphas[i]/alphas[i-1] for i in range(1, num_steps+1)]
            self.betas = torch.nn.Parameter(torch.tensor(betas, requires_grad=False))
        self.Q_matrix = torch.nn.Parameter(torch.zeros(num_steps, 2, 2, dtype=torch.float32), requires_grad=False)
        for i in range(num_steps):
            self.Q_matrix[i,:,:] = torch.tensor([[1-0.5*self.betas[i], 0.5*self.betas[i]], [0.5*self.betas[i], 1-0.5*self.betas[i]]], dtype=torch.float32)
        self.Q_prod_matrix = copy.deepcopy(self.Q_matrix)
        for i in range(1, num_steps):
            self.Q_prod_matrix[i,:,:] = torch.matmul(self.Q_prod_matrix[i-1,:,:], self.Q_prod_matrix[i,:,:])
        self.fiid = fiid
        self.embed_dim = embed_dim
        self.num_items = num_items
        self.scorenet = ScoreNet(embed_dim, num_items-1)

    def forward(self, batch):
        if self.training:
            output = self.diffusion_loss_func(batch["in_"+self.fiid])
        else:
            output = self.pc_sampler(batch["in_"+self.fiid])

        return output
    
    def diffusion_loss_func(self, item_ids):
        batch_size = item_ids.shape[0]

        x_int = torch.zeros([batch_size, self.num_items], dtype=torch.long, device=item_ids.device)
        x_int = x_int.scatter(1, item_ids, 1)
        x_int = x_int[:,1:] 
        x_one_hot  = torch.nn.functional.one_hot(x_int, num_classes=2).float()#(batch_size, num_items, 2)

        t = torch.randint(0, self.num_steps, size=(batch_size,), device=item_ids.device)

        Q_prod_t = self.Q_prod_matrix[t]#(batch_size, 2, 2)
        prob = torch.matmul(x_one_hot, Q_prod_t)#(batch_size, num_items, 2)
        x_perturbed = torch.distributions.Bernoulli(probs=prob[:,:,1]).sample()#(batch_size, num_items)

        output = self.scorenet(x_perturbed, t)
        #output += x_one_hot

        loss_func = nn.BCELoss(reduction='mean')
        pos_idx = torch.where(x_int>0)
        pos_loss = loss_func(output[pos_idx], x_int[pos_idx].float())
        neg_idx = torch.where(x_int==0)
        neg_loss = loss_func(output[neg_idx], x_int[neg_idx].float())
        
        return pos_loss + self.alpha * neg_loss

    def pc_sampler(self, item_ids):
        with torch.no_grad():
            cur_x = torch.distributions.Bernoulli(probs=0.5*torch.ones([item_ids.shape[0], self.num_items], device=item_ids.device)).sample()
            cur_x = cur_x.scatter(1, item_ids, 1)
            cur_x = cur_x[:,1:]
            for i in reversed(range(self.margin, self.num_steps, self.margin)):
                t = torch.tensor([i]*item_ids.shape[0], device=item_ids.device)
                x0_bar = self.scorenet(cur_x, t) # (batch_size, num_items)
                cur_x_one_hot = torch.nn.functional.one_hot(cur_x.long(), num_classes=2).float()#(batch_size, num_items, 2)
                x0_1 = torch.nn.functional.one_hot(torch.ones_like(cur_x, dtype=torch.long, device=item_ids.device), num_classes=2).float()#(batch_size, num_items, 2)
                x0_0 = torch.nn.functional.one_hot(torch.zeros_like(cur_x, dtype=torch.long, device=item_ids.device), num_classes=2).float()
                prob_1 = torch.matmul(cur_x_one_hot, self.Q_matrix[i].T) * torch.matmul(x0_1, self.Q_prod_matrix[i-self.margin]) \
                                / (torch.matmul(x0_1, self.Q_prod_matrix[i]) * cur_x_one_hot).sum(-1)[:,:,None]
                prob_0 = torch.matmul(cur_x_one_hot, self.Q_matrix[i].T) * torch.matmul(x0_0, self.Q_prod_matrix[i-self.margin]) \
                                / (torch.matmul(x0_0, self.Q_prod_matrix[i]) * cur_x_one_hot).sum(-1)[:,:,None]
                prob = x0_bar[:,:,None] * prob_1 + (1-x0_bar)[:,:,None] * prob_0
                cur_x = torch.argmax(prob, dim=-1).float()
        return prob[:, :, 1]
            



class D3PMVAE(BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('D3PMVAE')
        parent_parser.add_argument("--num_steps", type=int, default=1000, help="total diffusion steps")
        parent_parser.add_argument("--beta_min", type=float, default=0.02, help="min value for beta")
        parent_parser.add_argument("--beta_max", type=float, default=1.0, help="max value for beta")
        parent_parser.add_argument("--alpha", type=float, default=0.2, help="weight for negative samples loss")
        parent_parser.add_argument("--margin", type=int, default=1, help="DDIM margin")
        parent_parser.add_argument("--schedule", type=int, default=1, help="noise schedule")
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
        return D3PMVAEQueryEncoder(train_data.fiid, train_data.num_items, self.embed_dim, self.config['num_steps'], self.config['beta_min'], self.config['beta_max'],
                                    self.config['alpha'], self.config['margin'], self.config['schedule'])

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